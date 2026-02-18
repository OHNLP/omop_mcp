import asyncio
import logging
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mcp_use import MCPAgent, MCPClient

logger = logging.getLogger(__name__)

from omop_mcp import utils
from omop_mcp.prompts import EXAMPLE_INPUT, EXAMPLE_OUTPUT, MCP_DOC_INSTRUCTION

load_dotenv()

MAX_STEPS = 5
RECURSION_LIMIT = 25
REASONING_TIMEOUT = 15  # seconds for plain LLM reasoning step
AGENT_TIMEOUT = 30  # seconds for MCP agent tool-call step


def get_agent(
    llm_provider: str = "azure_openai",
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

    agent = MCPAgent(llm=llm, client=client, max_steps=MAX_STEPS)
    agent.recursion_limit = RECURSION_LIMIT
    return agent


def _extract_prior_context(history: list[dict[str, str]]) -> str:
    """Extract a concise context summary from the last assistant message.

    Parses the markdown table format produced by service.py to recover
    the concept name, domain, and vocabulary from the previous turn.
    Returns a short context string, or empty string if nothing found.
    """
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            break
    else:
        return ""

    # Extract bold concept name (e.g. **Type 2 diabetes mellitus**)
    name_match = re.search(r"\*\*(.+?)\*\*", content)
    if not name_match:
        return ""

    concept_name = name_match.group(1)

    # Extract domain and vocabulary from markdown table
    def _table_val(label: str) -> str:
        m = re.search(
            rf"\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|", content, re.IGNORECASE
        )
        return m.group(1).strip() if m else ""

    domain = _table_val("Domain")
    vocab = _table_val("Vocabulary")

    parts = [f"previous concept: `{concept_name}`"]
    if domain:
        parts.append(f"domain: {domain}")
    if vocab:
        parts.append(f"vocabulary: {vocab}")
    return ", ".join(parts)


async def run_agent(
    user_prompt: str,
    llm_provider: str = "azure_openai",
    llm=None,
    client=None,
    history: list[dict[str, str]] | None = None,
) -> dict:
    start_time = time.time()

    # Enrich the prompt with prior context for follow-up resolution
    prior_context = ""
    if history:
        prior_context = _extract_prior_context(history)
        if prior_context:
            # Prepend context directly to user_prompt so both steps see it
            user_prompt = f"{user_prompt} (Context: {prior_context})"
            logger.info(f"Enriched prompt with prior context: {prior_context}")

    history_context = ""
    if history:
        history_lines = []
        for msg in history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_context = (
            "Here is the conversation so far:\n"
            + "\n".join(history_lines)
            + "\n\nThe user's latest message is below. Use the conversation history "
            "to understand context and follow-up intent.\n\n"
        )

    reasoning_prompt = f"""{history_context}You are an OMOP concept mapping expert. Analyze this request and determine:
1. The user's INTENT — is this a concept mapping request, or a general question?
2. If mapping: what clinical keyword, OMOP table, and field to use.

    User request: "{user_prompt}"

    **First, determine the INTENT:**
    - "mapping" — user wants to find/map a specific OMOP concept. This includes:
      - Direct mapping requests (e.g., "aspirin", "map diabetes")
      - Follow-up requests that reference the previous concept (e.g., "non-standard version", "ICD10 code", "alternatives", "is there a non-standard concept")
      - ANY request that asks for a different version, vocabulary, or variant of a previously mapped concept is ALWAYS "mapping"
    - "general" — user is asking an educational/informational question with NO specific concept to look up (e.g., "what is OMOP?", "explain standard vs non-standard in general", "how does mapping work?")

    **If INTENT is mapping:**
    - Interpret the medical keyword clinically (expand abbreviations: CP→chest pain, BP→blood pressure, MI→myocardial infarction)
    - If this is a follow-up referencing a previous concept, use the context from the previous turn
    - Infer the OMOP table/field if not specified

    **If INTENT is general:**
    - Provide ANSWER directly — a clear, helpful response to the question
    - No need for KEYWORD/OMOP_TABLE/OMOP_FIELD/INFERRED_KEYWORD

    Output format:
    INTENT: [mapping or general]
    KEYWORD: [the clinical term to map — only if INTENT is mapping]
    OMOP_TABLE: [the OMOP table — only if INTENT is mapping]
    OMOP_FIELD: [the concept ID field — only if INTENT is mapping]
    INFERRED_KEYWORD: [what to actually search for in the vocabulary — only if INTENT is mapping]
    ANSWER: [direct response — only if INTENT is general]
    REASONING: [your interpretation and reasoning]
    """

    # Step 1: Plain LLM call for reasoning (no MCP tools needed)
    llm = utils.get_llm(provider=llm_provider) if llm is None else llm
    try:
        reasoning_result = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=reasoning_prompt)]),
            timeout=REASONING_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Reasoning step timed out after {REASONING_TIMEOUT}s")
    reasoning_response = (
        reasoning_result.content
        if hasattr(reasoning_result, "content")
        else str(reasoning_result)
    )
    logger.info("Reasoning step completed: keyword interpretation done")

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

    intent = _extract_field(reasoning_response, "INTENT").lower().strip()
    keyword = _extract_field(reasoning_response, "KEYWORD")
    omop_table = _extract_field(reasoning_response, "OMOP_TABLE")
    omop_field = _extract_field(reasoning_response, "OMOP_FIELD")
    inferred_keyword = _extract_field(reasoning_response, "INFERRED_KEYWORD")
    answer = _extract_field(reasoning_response, "ANSWER")

    # REASONING can be multi-line — capture everything after the label
    reasoning_match = re.search(
        r"(?:\*\*)?REASONING(?:\*\*)?:\s*([\s\S]+)", reasoning_response, re.IGNORECASE
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    processing_time = time.time() - start_time

    # If general question, return the LLM's answer directly — no tool call needed
    if intent == "general" and answer:
        logger.info("General question detected — skipping MCP tool call")
        return {
            "response": answer,
            "processing_time_sec": processing_time,
            "debug_info": {
                "intent": "general",
                "reasoning_step": reasoning_response,
            },
        }

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

**CRITICAL RULES:**
1. Call `find_omop_concept` at most TWICE. One primary search, and optionally one refinement if the first results are clearly wrong domain/vocabulary.
2. From the returned candidates, select the BEST match. Do NOT keep searching with more keyword variations.
3. If none of the candidates are a perfect match, pick the closest one and explain why in REASON.
4. After selecting your answer, immediately output the result in the required format below. Do NOT make any more tool calls.

The tool will return multiple candidate concepts. You must:
1. Review all candidates considering their Standard/Valid status, domain, vocabulary, and clinical appropriateness
2. Select the MOST APPROPRIATE concept based on the context and any specific requirements mentioned
3. Provide clear reasoning for your selection

{EXAMPLE_INPUT}

{EXAMPLE_OUTPUT}

After reviewing the candidates, provide your response in the exact format shown above, including the REASON field that explains your selection criteria and incorporates your keyword interpretation reasoning.
"""

    # Step 2: Create agent for tool calls (fresh instance, no leaked state)
    agent = get_agent(llm_provider, llm=llm, client=client)
    try:
        final_result = await asyncio.wait_for(
            agent.run(final_prompt),
            timeout=AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Agent tool-call step timed out after {AGENT_TIMEOUT}s")
    final_response = (
        final_result.content if hasattr(final_result, "content") else str(final_result)
    )

    processing_time = time.time() - start_time

    return {
        "response": final_response,
        "processing_time_sec": processing_time,
        "debug_info": {
            "intent": "mapping",
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
    llm_provider: str = "azure_openai",
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
