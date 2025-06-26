import asyncio
import os
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
    return await agent.run(query=prompt, external_history=history)


if __name__ == "__main__":

    async def test_mcp():
        prompt = "Map `Temperature Temporal Scanner - RR` for `measurement_concept_id` in the `measurement` table."
        result = await run_agent(prompt)
        print(result)

    asyncio.run(test_mcp())
