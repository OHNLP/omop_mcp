import asyncio
import os
import time
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from mcp_use import MCPAgent, MCPClient

from omop_mcp import utils
from omop_mcp.prompts import EXAMPLE_INPUT, EXAMPLE_OUTPUT, MCP_DOC_INSTRUCTION

load_dotenv()

MAX_STEPS = 5  # maximum number of steps for the agent


def get_agent(
    llm_provider: Literal["azure_openai", "openai"] = "azure_openai",
    llm=None,
    client=None,
) -> MCPAgent:
    if client is None:
        config_dir = os.path.dirname(__file__)
        config_path = os.path.join(config_dir, "../../omop_mcp_config.json")
        client = MCPClient.from_config_file(config_path)

    if llm is None:
        if llm_provider == "azure_openai":
            llm = AzureChatOpenAI(
                azure_deployment=os.getenv("MODEL_NAME"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                temperature=0,
                seed=42,
            )
        elif llm_provider == "openai":
            # Allow model override via env var, default to gpt-4o
            model_name = os.getenv("MODEL_NAME", "gpt-4o")
            llm = ChatOpenAI(model=model_name, temperature=0)
        elif llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            model_name = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
            llm = ChatAnthropic(model=model_name, temperature=0)
        elif llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        else:
            raise ValueError(
                f"Unsupported or unconfigured llm_provider: {llm_provider}. "
                "Pass an initialized 'llm' object or use one of: azure_openai, openai, anthropic, gemini."
            )

    return MCPAgent(llm=llm, client=client, max_steps=MAX_STEPS)


async def run_agent(
    user_prompt: str, llm_provider: str = "azure_openai", llm=None, client=None
) -> dict:
    start_time = time.time()

    agent = get_agent(llm_provider, llm=llm, client=client)

    # Step 1: Get LLM reasoning about keyword interpretation and extract components
    reasoning_prompt = f"""
    You are an OMOP concept mapping expert with deep clinical knowledge. Your task is to analyze this request and determine what the medical keyword actually means in clinical context, even if the user doesn't specify exact OMOP details.

    User request: "{user_prompt}"

    **CRITICAL: You must interpret the medical keyword clinically, not just extract it literally.**

    Consider:
    - What does this keyword mean in medical terminology?
    - What is the most likely clinical concept the user is looking for?
    - Are there common medical abbreviations that need expansion?
    - What would a clinician understand this to mean?
    - What OMOP table/field would be most appropriate if not specified?

    Examples of proper interpretation:
    - "CP" in condition context → "chest pain" (not just "CP")
    - "temp" in measurement context → "temperature" (not just "temp") 
    - "BP" in measurement context → "blood pressure" (not just "BP")
    - "MI" in condition context → "myocardial infarction" (not just "MI")

    **Handle natural language flexibly:**
    - "Map chest pain" → infer condition_occurrence.condition_concept_id
    - "Find concept for diabetes" → infer condition_occurrence.condition_concept_id
    - "What's the OMOP code for aspirin?" → infer drug_exposure.drug_concept_id
    - "Temperature measurement" → infer measurement.measurement_concept_id

    Output format:
    KEYWORD: [the main clinical term/keyword to map]
    OMOP_TABLE: [the OMOP table mentioned or implied - infer if not specified]
    OMOP_FIELD: [the OMOP field mentioned or implied - infer if not specified] 
    INFERRED_KEYWORD: [the keyword you would actually search for - this should be the CLINICAL interpretation, not the literal input]
    REASONING: [explain your clinical interpretation, why you expanded/changed the keyword, and how you inferred the OMOP table/field if not specified]

    **Remember: The INFERRED_KEYWORD should be what you would actually search for in a medical database, not the literal user input. If OMOP details aren't specified, make intelligent inferences based on the clinical concept.**
    """

    reasoning_result = await agent.run(reasoning_prompt)
    reasoning_response = (
        reasoning_result.content
        if hasattr(reasoning_result, "content")
        else str(reasoning_result)
    )

    # Parse the reasoning response
    keyword = ""
    omop_table = ""
    omop_field = ""
    inferred_keyword = ""
    reasoning = ""

    for line in reasoning_response.split("\n"):
        line = line.strip()
        if line.startswith("KEYWORD:"):
            keyword = line.replace("KEYWORD:", "").strip()
        elif line.startswith("OMOP_TABLE:"):
            omop_table = line.replace("OMOP_TABLE:", "").strip()
        elif line.startswith("OMOP_FIELD:"):
            omop_field = line.replace("OMOP_FIELD:", "").strip()
        elif line.startswith("INFERRED_KEYWORD:"):
            inferred_keyword = line.replace("INFERRED_KEYWORD:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    # If parsing failed, use fallbacks
    if not keyword:
        keyword = "unknown"
    if not inferred_keyword:
        inferred_keyword = keyword
    if not reasoning:
        reasoning = f"Used keyword '{inferred_keyword}' as provided."

    # Step 2: Use the extracted information in a tool call
    final_prompt = f"""
{MCP_DOC_INSTRUCTION}

Original user request: {user_prompt}

Based on your analysis, find concepts for `{inferred_keyword}` for `{omop_field}` in the `{omop_table}` table.

Your previous reasoning for this keyword was: {reasoning}

The tool will return multiple candidate concepts. You must:
1. Review all candidates considering their Standard/Valid status, domain, vocabulary, and clinical appropriateness
2. Select the MOST APPROPRIATE concept based on the context and any specific requirements mentioned
3. Provide clear reasoning for your selection

{EXAMPLE_INPUT}

{EXAMPLE_OUTPUT}

After reviewing the candidates, provide your response in the exact format shown above, including the REASON field that explains your selection criteria and incorporates your keyword interpretation reasoning.
"""

    # Let the agent handle the tool call
    final_result = await agent.run(final_prompt)
    final_response = (
        final_result.content if hasattr(final_result, "content") else str(final_result)
    )

    processing_time = time.time() - start_time

    return {
        "response": final_response,
        "processing_time_sec": processing_time,
        "debug_info": {
            "reasoning_step": reasoning_response,
            "extracted": {
                "keyword": keyword,
                "omop_table": omop_table,
                "omop_field": omop_field,
                "inferred_keyword": inferred_keyword,
            },
            "keyword_interpretation_reasoning": reasoning,
        },
    }


async def run_llm_no_mcp(
    prompt: str, llm_provider: Literal["azure_openai", "openai"] = "azure_openai"
):
    """
    Calls the LLM API directly with the provided prompt,
    using the same system message and few-shot example as the MCP agent.
    """
    load_dotenv()

    if llm_provider == "azure_openai":
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0,
            seed=42,
        )
    elif llm_provider == "openai":
        llm = ChatOpenAI(model="gpt-4o", temperature=0, seed=42)
    else:
        raise ValueError(
            f"Unsupported llm_provider: {llm_provider}. "
            "Valid options are 'azure_openai' or 'openai'."
        )

    messages = [
        SystemMessage(content=MCP_DOC_INSTRUCTION),
        HumanMessage(content=EXAMPLE_INPUT),
        AIMessage(content=EXAMPLE_OUTPUT),
        HumanMessage(content=prompt),
    ]

    start = time.perf_counter()
    response = await llm.ainvoke(messages)
    processing_time = time.perf_counter() - start

    return {"response": response.content, "processing_time_sec": processing_time}


if __name__ == "__main__":

    async def test_mcp():

        # prompt = "Map `Temperature Temporal Scanner - RR` for `measurement_concept_id` in the `measurement` table."

        prompt = "Map `Mean Arterial Pressure (Invasive)` for `measurement_concept_id` in the measurement` table"

        print("=" * 60)
        print("WITH MCP TOOLS:")
        print("=" * 60)
        mcp_result = await run_agent(prompt)
        print(mcp_result)

        print("\n" + "=" * 60)
        print("WITHOUT MCP TOOLS (LLM only):")
        print("=" * 60)
        no_mcp_result = await run_llm_no_mcp(prompt)
        print(no_mcp_result)

    asyncio.run(test_mcp())
