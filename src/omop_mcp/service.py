"""
OMOP MCP Service — High-level abstraction for mapping concepts via the Agent.
"""

import json
import os
import sys
import tempfile
import time
from typing import Any, Literal

from mcp_use import MCPClient

from omop_mcp import agent as omop_agent
from omop_mcp import utils as omop_utils

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
except ImportError:
    pass

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    pass


class MCPAgentService:
    def __init__(
        self,
        llm_provider: str = "azure",
        llm_api_key: str | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
        omophub_api_key: str | None = None,
    ):
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model

        # Ensure API key is available in env for potential subprocess/utils usage
        if omophub_api_key:
            os.environ["OMOPHUB_API_KEY"] = omophub_api_key

    def _get_mcp_config(self) -> dict:
        """
        Generate configuration for running the MCP server as a subprocess.
        """
        # Find the module path for omop_mcp to run it via python -m
        # This assumes omop_mcp is installed and importable in the current python env
        return {
            "mcpServers": {
                "omop_mcp": {
                    "command": sys.executable,
                    "args": ["-m", "omop_mcp"],
                    "env": {
                        # Pass through important environment variables
                        **{
                            k: v
                            for k, v in os.environ.items()
                            if k
                            in [
                                "OPENAI_API_KEY",
                                "ANTHROPIC_API_KEY",
                                "AZURE_OPENAI_API_KEY",
                                "OMOPHUB_API_KEY",
                                "PATH",
                                "SYSTEMROOT",
                                "HOME",
                            ]
                        }
                    },
                }
            }
        }

    def _get_llm(self):
        """
        Initialize the LangChain LLM object based on provider settings.
        """
        # Set API keys in env if provided, as some libraries prefer env vars
        if self.llm_api_key:
            if self.llm_provider == "openai":
                os.environ["OPENAI_API_KEY"] = self.llm_api_key
            elif self.llm_provider == "azure":
                os.environ["AZURE_OPENAI_API_KEY"] = self.llm_api_key
            elif self.llm_provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.llm_api_key
            elif self.llm_provider == "gemini":
                os.environ["GOOGLE_API_KEY"] = self.llm_api_key

        if self.llm_provider == "azure":
            return AzureChatOpenAI(
                azure_deployment=self.llm_model or os.getenv("MODEL_NAME", "gpt-4o"),
                azure_endpoint=self.llm_endpoint
                or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=self.llm_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
                temperature=0,
            )
        elif self.llm_provider == "openai":
            return ChatOpenAI(
                model=self.llm_model or "gpt-4o",
                api_key=self.llm_api_key or os.getenv("OPENAI_API_KEY", ""),
                temperature=0,
            )
        elif self.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.llm_model or "claude-sonnet-4-20250514",
                api_key=self.llm_api_key or os.getenv("ANTHROPIC_API_KEY", ""),
                temperature=0,
            )
        elif self.llm_provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.llm_model or "gemini-2.5-flash",
                api_key=self.llm_api_key or os.getenv("GOOGLE_API_KEY", ""),
                temperature=0,
            )
        elif self.llm_provider == "ollama":
            base = (self.llm_endpoint or "http://localhost:11434").rstrip("/")
            return ChatOpenAI(
                model=self.llm_model or "llama3",
                base_url=f"{base}/v1",
                api_key="ollama",
                temperature=0,
            )
        else:
            raise ValueError(
                f"Unsupported llm_provider: '{self.llm_provider}'. "
                "Use one of: azure, openai, anthropic, gemini, ollama."
            )

    async def map_concept(
        self, user_message: str, context: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Orchestrate the mapping process:
        1. Construct prompt with context
        2. Configure MCP client
        3. Initialize LLM
        4. Run Agent
        5. Parse results
        """
        start_time = time.time()

        # Prepare context if provided
        prompt = user_message
        if context:
            table = context.get("omop_table", "")
            field = context.get("omop_field", "")
            if table and field:
                prompt += f" for `{field}` in the `{table}` table"

        # Create temporary config file for MCP client
        config = self._get_mcp_config()
        config_fd, config_path = tempfile.mkstemp(suffix=".json")

        client = None
        try:
            with os.fdopen(config_fd, "w") as f:
                json.dump(config, f)

            # Create client from our temp config
            client = MCPClient.from_config_file(config_path)
            llm = self._get_llm()

            # Run the agent logic from the package, passing our custom components
            result = await omop_agent.run_agent(
                prompt, llm_provider=self.llm_provider, llm=llm, client=client
            )

            response_text = result.get("response", "")

            # Parse using omop_mcp.utils
            parsed = omop_utils.parse_agent_response(response_text)

            # Adapt to frontend format (generic structure)
            concepts = []
            if parsed.get("concept_id"):
                try:
                    cid = int(parsed.get("concept_id"))
                except (ValueError, TypeError):
                    cid = 0

                concepts.append(
                    {
                        "concept_id": cid,
                        "concept_name": parsed.get("name"),
                        "vocabulary_id": parsed.get("vocab"),
                        "domain_id": parsed.get("domain"),
                        "reasoning": parsed.get("reason"),
                        "confidence": "high",
                    }
                )

            processing_time = time.time() - start_time
            return {
                "message": response_text,
                "concepts": concepts,
                "processing_time": processing_time,
                "debug_info": result.get("debug_info"),
            }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "message": f"MCP Agent Error: {str(e)}",
                "concepts": [],
                "processing_time": processing_time,
            }
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if client:
                await client.close_all_sessions()
