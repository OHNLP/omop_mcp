import asyncio
import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from mcp_use import MCPAgent, MCPClient

from omop_mcp.prompts import EXAMPLE_INPUT, EXAMPLE_OUTPUT, MCP_DOC_INSTRUCTION

load_dotenv()


def get_agent(
    llm_provider: Literal["azure_openai", "openai"] = "azure_openai",
) -> MCPAgent:
    """
    Create an MCPAgent using the specified LLM provider and register the find_omop_concept tool.
    """
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, "../../omop_mcp_config.json")
    client = MCPClient.from_config_file(config_path)

    if llm_provider == "azure_openai":
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_WEST"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY_WEST"),
            api_version=os.getenv("AZURE_API_VERSION_WEST"),
        )
    elif llm_provider == "openai":
        llm = ChatOpenAI(model="gpt-4o")
    else:
        raise ValueError(
            f"Unsupported llm_provider: {llm_provider}. "
            "Valid options are 'azure_openai' or 'openai'."
        )

    return MCPAgent(llm=llm, client=client, max_steps=30)


def ensure_processing_time_in_output(response: str, processing_time: str) -> str:
    """
    Ensure the processing time is included in the response in the correct location.
    Remove duplicates and ensure only one properly formatted processing time entry.
    """
    lines = response.split("\n")

    # Remove any processing time mentions (both structured and explanatory)
    cleaned_lines = []
    url_index = -1

    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Skip lines that mention processing time in any format
        if (
            "processing time" in line_lower
            or "**processing_time_sec**" in line
            or "processing_time_sec:" in line
            or (
                "processing" in line_lower
                and ("seconds" in line_lower or "sec" in line_lower)
            )
        ):
            continue

        # Track URL position for insertion
        if "**url**" in line.lower() or "athena.ohdsi.org" in line:
            url_index = len(cleaned_lines)  # Position after this line

        cleaned_lines.append(line)

    # Add the processing time in the correct location (after URL) only if we have a valid time
    if processing_time is not None:
        if url_index >= 0:
            cleaned_lines.insert(
                url_index + 1, f"- **PROCESSING_TIME_SEC**: {processing_time}"
            )
        else:
            # Fallback: add before any explanatory text
            insert_index = len(cleaned_lines)
            for i, line in enumerate(cleaned_lines):
                if line.strip().startswith("This concept") or line.strip().startswith(
                    "The "
                ):
                    insert_index = i
                    break
            cleaned_lines.insert(
                insert_index, f"- **PROCESSING_TIME_SEC**: {processing_time}"
            )
    else:
        # If processing time is None, add a note indicating it couldn't be captured
        if url_index >= 0:
            cleaned_lines.insert(url_index + 1, "- **PROCESSING_TIME_SEC**: None")
        else:
            # Fallback: add before any explanatory text
            insert_index = len(cleaned_lines)
            for i, line in enumerate(cleaned_lines):
                if line.strip().startswith("This concept") or line.strip().startswith(
                    "The "
                ):
                    insert_index = i
                    break
            cleaned_lines.insert(insert_index, "- **PROCESSING_TIME_SEC**: None")

    return "\n".join(cleaned_lines)


def get_real_processing_time() -> str:
    """Get the real processing time from the server module."""
    try:
        from omop_mcp import server

        if hasattr(server, "_last_processing_time") and server._last_processing_time:
            return server._last_processing_time
    except:
        pass
    return None


def extract_processing_time_from_response(response: str) -> str:
    """Extract processing time from the LLM response as a fallback."""
    patterns = [
        r"approximately (\d+\.?\d*) seconds?",
        r"(\d+\.?\d*) seconds?",
        r"took (\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    return "0.000"


def clean_url_formatting(response: str) -> str:
    """
    Remove markdown formatting from URLs in the response.
    Converts [text](url) format to just the plain URL.
    """
    import re

    # Pattern to match markdown links
    markdown_link_pattern = r"\[([^\]]+)\]\((https://athena\.ohdsi\.org/[^)]+)\)"

    def replace_markdown_link(match):
        url = match.group(2)
        return url

    return re.sub(markdown_link_pattern, replace_markdown_link, response)


def clean_no_mcp_response(response: str, actual_time: str) -> str:
    """
    Clean up the response from LLM-only mode by replacing the processing time
    with the actual measured time for the LLM API call.
    """
    import re

    # Replace any processing time with the actual measured time
    response = re.sub(
        r"PROCESSING_TIME_SEC:\s*[\d.]+",
        f"PROCESSING_TIME_SEC: {actual_time}",
        response,
    )

    return response


async def run_agent(
    prompt: str, llm_provider: Literal["azure_openai", "openai"] = "azure_openai"
):
    """
    Run the MCP agent with the given prompt and LLM provider.
    """
    agent = get_agent(llm_provider=llm_provider)
    history = [
        SystemMessage(content=MCP_DOC_INSTRUCTION),
        HumanMessage(content=EXAMPLE_INPUT),
        AIMessage(content=EXAMPLE_OUTPUT),
    ]

    # Get the response from the agent
    response = await agent.run(query=prompt, external_history=history)

    # Get the real processing time
    processing_time = get_real_processing_time()

    # Fallback: extract from response if global variable doesn't work
    if processing_time is None and isinstance(response, str):
        extracted_time = extract_processing_time_from_response(response)
        if extracted_time != "0.000":
            processing_time = extracted_time

    # Ensure processing time is in the response and clean URL formatting
    if isinstance(response, str):
        response = ensure_processing_time_in_output(response, processing_time)
        response = clean_url_formatting(response)

    return response


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
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_WEST"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY_WEST"),
            api_version=os.getenv("AZURE_API_VERSION_WEST"),
        )
    elif llm_provider == "openai":
        llm = ChatOpenAI(model="gpt-4o")
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
    elapsed = time.perf_counter() - start

    cleaned_response = clean_no_mcp_response(response.content, f"{elapsed:.3f}")
    return cleaned_response


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
