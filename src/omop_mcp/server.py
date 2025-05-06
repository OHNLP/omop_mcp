import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import mcp.types as types
from mcp.server.fastmcp import FastMCP

# Get absolute path to the data file
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "omop_concept_id_fields.json"

# Load OMOP CDM table/field mapping from JSON file
with open(DATA_FILE, "r") as f:
    OMOP_CDM = json.load(f)

MCP_DOC_INSTRUCTION = (
    "When selecting the best OMOP concept and vocabulary, always refer to the "
    "official OMOP CDM v5.4 documentation: "
    "https://ohdsi.github.io/CommonDataModel/faq.html and "
    "https://ohdsi.github.io/CommonDataModel/vocabulary.html. "
    "Use the mapping conventions, standard concept definitions, and vocabulary "
    "guidance provided there to ensure your selection is accurate and consistent "
    "with OMOP best practices. Prefer concepts that are marked as 'Standard' and "
    "'Valid', and use the recommended vocabularies for each domain (e.g., SNOMED "
    "for conditions, RxNorm for drugs, LOINC for measurements)."
)


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize HTTP session
    async with aiohttp.ClientSession() as session:
        yield {"http_session": session}


# Initialize server with lifespan
mcp = FastMCP("omop_concept_mapper", lifespan=server_lifespan)


@mcp.resource("omop://tables")
async def list_omop_tables() -> Dict[str, List[str]]:
    """List all OMOP CDM tables and their concept ID fields."""
    return OMOP_CDM


@mcp.prompt()
async def map_clinical_concept() -> types.GetPromptResult:
    """Create a prompt for mapping clinical concepts."""
    return types.GetPromptResult(
        description="Map a clinical term to OMOP concept",
        messages=[
            types.PromptMessage(
                role="system",
                content=types.TextContent(type="text", text=MCP_DOC_INSTRUCTION),
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Please help map this clinical term to an OMOP concept: {{term}}",
                ),
            ),
        ],
    )


@mcp.tool()
async def find_omop_concept(
    keyword: str, omop_table: str, omop_field: str
) -> Dict[str, Any]:
    """
    Find the best-matching OMOP concept for a given keyword, table, and field.

    Args:
        keyword: The clinical term to map
        omop_table: The OMOP CDM table name
        omop_field: The concept ID field name

    Returns:
        Dict containing the best matching concept or error information
    """
    # Validate OMOP table and field
    if omop_table not in OMOP_CDM:
        return {
            "error": f"OMOP table '{omop_table}' not found in OMOP CDM.",
            "instruction": MCP_DOC_INSTRUCTION,
        }
    if omop_field not in OMOP_CDM[omop_table]:
        return {
            "error": f"Field '{omop_field}' not found in OMOP table '{omop_table}'.",
            "instruction": MCP_DOC_INSTRUCTION,
        }

    # Get HTTP session from lifespan context
    ctx = mcp.request_context
    session: aiohttp.ClientSession = ctx.lifespan_context["http_session"]

    url = "https://athena.ohdsi.org/api/v1/concepts"
    params = {"query": keyword}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/111.0.0.0 Safari/537.36"
        )
    }

    try:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
    except aiohttp.ClientError as e:
        return {
            "error": f"Failed to query Athena: {str(e)}",
            "instruction": MCP_DOC_INSTRUCTION,
        }

    concepts = []
    if isinstance(data, list):
        concepts = data
    elif isinstance(data, dict):
        for key in ("content", "results", "items", "concepts"):
            if key in data and isinstance(data[key], list):
                concepts = data[key]
                break

    if not concepts:
        return {
            "error": "No results found or unexpected response structure.",
            "instruction": MCP_DOC_INSTRUCTION,
        }

    # Prioritize Standard and Valid concepts
    prioritized = []
    for c in concepts:
        is_standard = (
            c.get("standardConcept", "").lower() == "standard"
            or c.get("standardConcept", "").upper() == "S"
        )
        is_valid = c.get("validity", c.get("invalidReason", "")).lower() == "valid"
        if is_standard and is_valid:
            prioritized.append(c)

    if not prioritized:
        prioritized = concepts

    # LLM-based reasoning placeholder: just return the first for now
    best = prioritized[0]

    return {
        "id": best.get("id", ""),
        "code": best.get("code", ""),
        "name": best.get("name", ""),
        "class": best.get("classId", best.get("className", "")),
        "concept": best.get("standardConcept", ""),
        "validity": best.get("validity", best.get("invalidReason", "")),
        "domain": best.get("domain", best.get("domainId", "")),
        "vocab": best.get("vocabulary", best.get("vocabularyId", "")),
        "instruction": MCP_DOC_INSTRUCTION,
    }


if __name__ == "__main__":
    mcp.run()
