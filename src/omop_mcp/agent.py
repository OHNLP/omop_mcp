import asyncio
import os
import re
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

    # Add the processing time in the correct location (after URL)
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

    return "\n".join(cleaned_lines)


def get_real_processing_time() -> str:
    """Get the real processing time from the server module."""
    try:
        from omop_mcp import server

        if hasattr(server, "_last_processing_time") and server._last_processing_time:
            return server._last_processing_time
    except:
        pass
    return "0.000"


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
        text = match.group(1)
        url = match.group(2)
        return url

    return re.sub(markdown_link_pattern, replace_markdown_link, response)


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
    if processing_time == "0.000" and isinstance(response, str):
        extracted_time = extract_processing_time_from_response(response)
        if extracted_time != "0.000":
            processing_time = extracted_time

    # Ensure processing time is in the response and clean URL formatting
    if isinstance(response, str):
        response = ensure_processing_time_in_output(response, processing_time)
        response = clean_url_formatting(response)

    return response


if __name__ == "__main__":

    async def test_mcp():
        prompt = "Map `Temperature Temporal Scanner - RR` for `measurement_concept_id` in the `measurement` table."
        result = await run_agent(prompt)
        print(result)

    asyncio.run(test_mcp())
