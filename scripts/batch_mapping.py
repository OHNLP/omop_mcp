import asyncio
from pathlib import Path

import pandas as pd

from omop_mcp import utils
from omop_mcp.agent import run_agent


async def main(llm_provider, data_path, batch_size):
    # Set data path
    data_path = Path(data_path)
    result_path = data_path.with_suffix("_results.csv")

    # Load data
    df = pd.read_csv(data_path)

    if result_path.exists():
        existing_results = pd.read_csv(result_path)
        processed_keywords = set(existing_results["keyword"])
        print(f"Found {len(existing_results)} existing results")
    else:
        existing_results = pd.DataFrame()
        processed_keywords = set()
        print("No existing results found, starting from beginning")

    # Filter df to only unprocessed rows
    df = df[~df["keyword"].isin(processed_keywords)]
    print(f"Remaining rows to process: {len(df)}")

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        results = []
        for _, row in batch_df.iterrows():
            keyword = row["keyword"].strip()
            prompt = f"Map `{keyword}` for `{row['omop_field']}` in the `{row['omop_table']}` table."

            print(f"Processing {keyword}...")
            agent_result = await run_agent(prompt, llm_provider=llm_provider)
            print(agent_result)
            llm_response = utils.parse_agent_response(agent_result["response"])

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

            if llm_response["concept_id"] and llm_response["name"]:
                athena_name = utils.get_concept_name_from_athena(
                    llm_response["concept_id"]
                )
                if athena_name is not None:
                    result["names_match"] = (
                        athena_name.lower() == llm_response["name"].lower()
                    )
                else:
                    result["names_match"] = None
            else:
                result["names_match"] = None
            results.append(result)

        batch_results = pd.DataFrame(results)

        if len(existing_results) == 0 and batch_start == 0:
            batch_results.to_csv(result_path, index=False)
        else:
            batch_results.to_csv(result_path, mode="a", header=False, index=False)

        print(f"Batch {batch_start//batch_size + 1} appended.")


if __name__ == "__main__":

    # Set parameters here
    DATA_PATH = "../data/evaluate/batch_mapping.csv"
    LLM_PROVIDER = "azure_openai"
    BATCH_SIZE = 50

    asyncio.run(main(LLM_PROVIDER, DATA_PATH, BATCH_SIZE))
