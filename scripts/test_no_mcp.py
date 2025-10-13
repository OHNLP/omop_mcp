import asyncio
from pathlib import Path

import pandas as pd

from omop_mcp import utils
from omop_mcp.agent import run_llm_no_mcp

# Test data
data_path = Path("../data/evaluate/combined_mapping.csv")
df = pd.read_csv(data_path)


async def main():
    # Run mapping agent without MCP tools
    results = []
    for _, row in df.iterrows():
        keyword = row["keyword"].strip()
        prompt = f"Map `{keyword}` for `{row['omop_field']}` in the `{row['omop_table']}` table."

        print(f"Processing {keyword} (NO MCP)...")
        no_mcp_result = await run_llm_no_mcp(prompt)
        print(no_mcp_result)

        # Try to parse the response (might fail since format could be different)
        try:
            llm_response = utils.parse_agent_response(no_mcp_result["response"])
        except Exception as e:
            print(f"Failed to parse response for {keyword}: {e}")
            # Create a fallback response structure
            llm_response = {
                "concept_id": None,
                "code": None,
                "name": None,
                "class": None,
                "concept": None,
                "validity": None,
                "domain": None,
                "vocab": None,
                "url": None,
                "reason": "Failed to parse response",
            }

        # Combine all results
        result = {
            "keyword": row["keyword"],
            "omop_field": row["omop_field"],
            "omop_table": row["omop_table"],
            "concept_id": llm_response["concept_id"],
            "code": llm_response["code"],
            "concept_name": llm_response["name"],
            "class": llm_response["class"],
            "concept": llm_response["concept"],
            "validity": llm_response["validity"],
            "domain": llm_response["domain"],
            "vocab": llm_response["vocab"],
            "url": llm_response["url"],
            "processing_time_sec": no_mcp_result["processing_time_sec"],
            "reason": llm_response["reason"],
            "concept_exists": (
                utils.concept_id_exists_in_athena(llm_response["concept_id"])
                if llm_response["concept_id"]
                else False
            ),
        }

        # Handle names_match separately to avoid NoneType errors
        if llm_response["concept_id"] and llm_response["name"]:
            athena_name = utils.get_concept_name_from_athena(llm_response["concept_id"])
            if athena_name is not None:
                result["names_match"] = (
                    athena_name.lower() == llm_response["name"].lower()
                )
            else:
                result["names_match"] = None
        else:
            result["names_match"] = None
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"./tests/data/{data_path.stem}_results_no_mcp.csv", index=False)
    print(df_results)
    return df_results


if __name__ == "__main__":
    asyncio.run(main())
