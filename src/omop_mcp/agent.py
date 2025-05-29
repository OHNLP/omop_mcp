import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


def get_agent():
    load_dotenv()
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, "../../omop_mcp_config.json")
    client = MCPClient.from_config_file(config_path)
    llm = ChatOpenAI(model="gpt-4o")
    return MCPAgent(llm=llm, client=client, max_steps=30)


async def run_agent(prompt: str):
    agent = get_agent()
    return await agent.run(prompt)


async def main():
    load_dotenv()

    # Use config file for local OMOP MCP server
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, "../../omop_mcp_config.json")
    client = MCPClient.from_config_file(config_path)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    result = await agent.run(
        "Map `Temperature Temporal Scanner - RR` for "
        "`measurement_concept_id` in the `measurement` table."
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
