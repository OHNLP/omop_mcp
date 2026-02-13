import asyncio
import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mcp_use import MCPAgent, MCPClient

from omop_mcp import utils
from omop_mcp.prompts import EXAMPLE_INPUT, EXAMPLE_OUTPUT, MCP_DOC_INSTRUCTION

load_dotenv()

MAX_STEPS = 5  # maximum number of steps for the agent


def get_agent(
    llm_provider: Literal[
        "azure_openai", "openai", "anthropic", "gemini", "ollama"
    ] = "azure_openai",
    llm=None,
    client=None,
) -> MCPAgent:
    if client is None:
        # Use simple relative path or environment variable for config
        # Assuming we are running from src/omop_mcp usually
        config_path = os.getenv("MCP_CONFIG_PATH")
        if not config_path:
            # Fallback to looking relative to this file
            config_dir = os.path.dirname(__file__)
            config_path = os.path.join(config_dir, "../../omop_mcp_config.json")

        if os.path.exists(config_path):
            client = MCPClient.from_config_file(config_path)
        else:
            pass

    if llm is None:
        llm = utils.get_llm(provider=llm_provider)

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

    # Parse the reasoning response using regex (handles markdown bold, varied casing, etc.)
    def _extract_field(text, field_name):
        patterns = [
            rf"\*\*{field_name}\*\*:\s*(.+)",
            rf"{field_name}:\s*(.+)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    keyword = _extract_field(reasoning_response, "KEYWORD")
    omop_table = _extract_field(reasoning_response, "OMOP_TABLE")
    omop_field = _extract_field(reasoning_response, "OMOP_FIELD")
    inferred_keyword = _extract_field(reasoning_response, "INFERRED_KEYWORD")

    # REASONING can be multi-line — capture everything after the label
    reasoning_match = re.search(
        r"(?:\*\*)?REASONING(?:\*\*)?:\s*([\s\S]+)", reasoning_response, re.IGNORECASE
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

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
    prompt: str,
    llm_provider: Literal[
        "azure_openai", "openai", "anthropic", "gemini", "ollama"
    ] = "azure_openai",
):
    """
    Calls the LLM API directly with the provided prompt,
    using the same system message and few-shot example as the MCP agent.
    """
    load_dotenv()

    llm = utils.get_llm(provider=llm_provider, seed=42)

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

        prompt = "Map `Mean Arterial Pressure (Invasive)` for `measurement_concept_id` in the `measurement` table"

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
