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
            temperature=0,
            seed=42,
        )
    elif llm_provider == "openai":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        raise ValueError(
            f"Unsupported llm_provider: {llm_provider}. "
            "Valid options are 'azure_openai' or 'openai'."
        )

    return MCPAgent(llm=llm, client=client, max_steps=30)


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
    start = time.perf_counter()
    response = await agent.run(query=prompt, external_history=history)
    elapsed = time.perf_counter() - start

    if isinstance(response, str):
        response = clean_url_formatting(response)

    return {"response": response, "elapsed": elapsed}


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
    elapsed = time.perf_counter() - start

    return {"response": response.content, "elapsed": elapsed}


if __name__ == "__main__":

    async def test_mcp():

        # prompt = "Map `Temperature Temporal Scanner - RR` for `measurement_concept_id` in the `measurement` table."

        prompt = "Map `Mean Arterial Pressure (Invasive)` for `measurement_concept_id` in the measurement` table"

        print("=" * 60)
        print("WITH MCP TOOLS:")
        print("=" * 60)
        mcp_result = await run_agent(prompt)
        print(mcp_result)

        # print("\n" + "=" * 60)
        # print("WITHOUT MCP TOOLS (LLM only):")
        # print("=" * 60)
        # no_mcp_result = await run_llm_no_mcp(prompt)
        # print(no_mcp_result)

    asyncio.run(test_mcp())
