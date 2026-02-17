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
from omop_mcp import config as omop_config
from omop_mcp import utils as omop_utils


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
                            if k in omop_config.ENV_VARS
                        }
                    },
                }
            }
        }

    def _get_llm(self):
        """
        Initialize the LangChain LLM object using centralized utils.
        """
        return omop_utils.get_llm(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.llm_api_key,
            endpoint=self.llm_endpoint,
        )

    async def map_concept(
        self,
        user_message: str,
        context: dict[str, str] | None = None,
        history: list[dict[str, str]] | None = None,
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
                prompt,
                llm_provider=self.llm_provider,
                llm=llm,
                client=client,
                history=history,
            )

            response_text = result.get("response", "")

            # Parse using omop_mcp.utils
            parsed = omop_utils.parse_agent_response(response_text)

            # Adapt to frontend format (generic structure)
            concepts = []
            friendly_message = response_text  # fallback to raw
            reasoning_text = None

            if parsed.get("concept_id"):
                try:
                    cid = int(parsed.get("concept_id"))
                except (ValueError, TypeError):
                    cid = 0

                reason = parsed.get("reason", "")
                concepts.append(
                    {
                        "concept_id": cid,
                        "concept_name": parsed.get("name"),
                        "vocabulary_id": parsed.get("vocab"),
                        "domain_id": parsed.get("domain"),
                        "reasoning": reason,
                        "confidence": "high",
                    }
                )

                name = parsed.get("name", "")
                vocab = parsed.get("vocab", "")
                domain = parsed.get("domain", "")
                concept_class = parsed.get("class", "")
                code = parsed.get("code", "")
                validity = parsed.get("validity", "")
                concept_type = parsed.get("concept", "")
                url = parsed.get("url", "")

                friendly_message = (
                    f"**{name}**\n\n"
                    f"| Field | Value |\n"
                    f"|-------|-------|\n"
                    f"| Concept ID | {cid} |\n"
                    f"| Code | {code} |\n"
                    f"| Vocabulary | {vocab} |\n"
                    f"| Domain | {domain} |\n"
                    f"| Class | {concept_class} |\n"
                    f"| Type | {concept_type} |\n"
                    f"| Validity | {validity} |\n"
                    f"| Athena | {url} |"
                )

                reasoning_text = reason

            processing_time = time.time() - start_time
            return {
                "message": friendly_message,
                "reasoning": reasoning_text,
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
