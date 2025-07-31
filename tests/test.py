import asyncio
from pathlib import Path

import pandas as pd

from omop_mcp import utils
from omop_mcp.agent import run_agent

# Test data
data_path = Path("../data/evaluate/combined_mapping.csv")
df = pd.read_csv(data_path)


async def main():
    # Run mapping agent
    results = []
    for _, row in df.iterrows():
        keyword = row["keyword"].strip()
        prompt = f"Map `{keyword}` for `{row['omop_field']}` in the `{row['omop_table']}` table."

        print(f"Processing {keyword}...")
        agent_result = await run_agent(prompt)
        print(agent_result)
        llm_response = utils.parse_agent_response(agent_result["response"])

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
            "processing_time_sec": agent_result["processing_time_sec"],
            "reason": llm_response["reason"],
            "concept_exists": utils.concept_id_exists_in_athena(
                llm_response["concept_id"]
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
    df_results.to_csv(f"./tests/data/{data_path.stem}_results.csv", index=False)
    print(df_results)
    return df_results


if __name__ == "__main__":
    asyncio.run(main())
