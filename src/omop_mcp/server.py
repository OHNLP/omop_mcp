import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import mcp.types as types
from mcp.server.fastmcp import FastMCP

from omop_mcp.prompts import EXAMPLE_INPUT, EXAMPLE_OUTPUT, MCP_DOC_INSTRUCTION

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "omop_concept_id_fields.json"

# Load OMOP CDM table/field mapping from JSON file
with open(DATA_FILE, "r") as f:
    OMOP_CDM = json.load(f)

# Global variable to store the last processing time
_last_processing_time = None

# Initialize server
mcp = FastMCP(name="omop_concept_mapper")


@mcp.resource("omop://tables")
async def list_omop_tables() -> Dict[str, List[str]]:
    """List all OMOP CDM tables and their concept ID fields."""
    return OMOP_CDM


@mcp.tool()
async def get_last_processing_time() -> Dict[str, Any]:
    """Get the processing time from the last find_omop_concept call."""
    global _last_processing_time
    return {"last_processing_time_sec": _last_processing_time}


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
                    text=EXAMPLE_INPUT,
                ),
            ),
            types.PromptMessage(
                role="assistant",
                content=types.TextContent(type="text", text=EXAMPLE_OUTPUT),
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="{{term}}",
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
        Dict containing the best matching concept or error information.
        Processing time is only returned on success.
    """
    global _last_processing_time

    logging.info(
        f"find_omop_concept called with keyword='{keyword}', omop_table='{omop_table}', omop_field='{omop_field}'"
    )

    start = time.perf_counter()

    # Create a new session for each request
    async with aiohttp.ClientSession() as session:
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
            return {
                "error": "No 'Standard' and 'Valid' concept found for the given keyword.",
            }

        best = prioritized[0]
        elapsed = time.perf_counter() - start

        # Store the processing time globally
        _last_processing_time = f"{elapsed:.3f}"

        logging.info(f"elapsed: {elapsed}")

        return {
            "concept_id": best.get("id", ""),
            "code": best.get("code", ""),
            "name": best.get("name", ""),
            "class": best.get("classId", best.get("className", "")),
            "concept": best.get("standardConcept", ""),
            "validity": best.get("validity", best.get("invalidReason", "")),
            "domain": best.get("domain", best.get("domainId", "")),
            "vocab": best.get("vocabulary", best.get("vocabularyId", "")),
            "url": f"https://athena.ohdsi.org/search-terms/terms/{best.get('id', '')}",
            "processing_time_sec": f"{elapsed:.3f}",
        }


@mcp.tool()
async def batch_map_concepts_from_csv(csv_path: str) -> str:
    """
    Process a CSV file of keywords, mapping each row and returning a CSV with mapping results appended as new columns.
    """
    output = io.StringIO()
    with open(csv_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [
            "concept_id",
            "code",
            "name",
            "class",
            "concept",
            "validity",
            "domain",
            "vocab",
            "url",
            "processing_time_sec",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            keyword = row.get("keywords", "")
            omop_field = row.get("omop_field", "")
            omop_table = row.get("omop_table", "")
            result = await find_omop_concept(keyword, omop_table, omop_field)
            # Merge result into row
            row.update(result)
            writer.writerow(row)
    return output.getvalue()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
