import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
import aiohttp
import mcp.types as types
from mcp.server.fastmcp import FastMCP

from omop_mcp import utils
from omop_mcp.prompts import EXAMPLE_INPUT, EXAMPLE_OUTPUT, MCP_DOC_INSTRUCTION

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "omop_concept_id_fields.json"

# Load OMOP CDM table/field mapping from JSON file
with open(DATA_FILE, "r") as f:
    OMOP_CDM = json.load(f)

# Initialize server
mcp = FastMCP(name="omop_concept_mapper")


@mcp.resource("omop://tables")
async def list_omop_tables() -> Dict[str, List[str]]:
    """List all OMOP CDM tables and their concept ID fields."""
    return OMOP_CDM


@mcp.resource("omop://documentation")
async def omop_documentation() -> str:
    """Fetch live OMOP CDM documentation including vocabulary rules."""
    url = "https://ohdsi.github.io/CommonDataModel/vocabulary.html"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract main content
                main_content = (
                    soup.find("div", class_="container-fluid main-container")
                    or soup.body
                )

                if main_content:
                    text = main_content.get_text()
                    # Clean up
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (
                        phrase.strip() for line in lines for phrase in line.split("  ")
                    )
                    clean_text = " ".join(chunk for chunk in chunks if chunk)
    return clean_text


@mcp.resource("omop://preferred_vocabularies")
async def get_vocabulary_preference() -> Dict[str, List[str]]:
    """Preferred vocabulary for each OMOP domain in the order of preference."""
    return {
        "measurement": ["LOINC", "SNOMED"],
        "condition_occurrence": ["SNOMED", "ICD10CM", "ICD9CM"],
        "drug_exposure": ["RxNorm", "RxNorm Extension", "SNOMED"],
        "procedure_occurrence": ["SNOMED", "CPT4", "ICD10PCS"],
        "observation": ["SNOMED"],
        "device_exposure": ["SNOMED"],
    }


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
    keyword: str, omop_table: str, omop_field: str, max_results: int = 20
) -> Dict[str, Any]:
    """
    Find OMOP concepts for a given keyword, table, and field.
    Returns multiple candidates for LLM to choose from based on context.

    Args:
        keyword: The clinical term to map
        omop_table: The OMOP CDM table name
        omop_field: The concept ID field name
        max_results: Maximum number of candidate concepts to return

    Returns:
        Dict containing candidate concepts or error information if no results found.
    """
    logging.info(
        f"find_omop_concept called with keyword='{keyword}', omop_table='{omop_table}', omop_field='{omop_field}'"
    )

    try:
        concepts = await utils.search_omophub_concepts_async(keyword, max_results)
    except Exception as e:
        logging.error(f"OMOPHub API call failed: {e}")
        raise RuntimeError(f"OMOPHub API is not accessible: {e}") from e

    if not concepts:
        return {
            "error": f"No results found for keyword '{keyword}'. The search term may not exist in the OMOP vocabulary.",
        }

    candidates = []
    for c in concepts[:max_results]:
        cid = c.get("concept_id", "")
        candidate = {
            "concept_id": cid,
            "code": c.get("concept_code", ""),
            "name": c.get("concept_name", ""),
            "class": c.get("concept_class_id", ""),
            "concept": c.get("standard_concept", ""),
            "validity": c.get("invalid_reason", "Valid"),
            "domain": c.get("domain_id", ""),
            "vocab": c.get("vocabulary_id", ""),
            "url": f"https://athena.ohdsi.org/search-terms/terms/{cid}",
        }
        candidates.append(candidate)

    return {
        "candidates": candidates,
        "search_metadata": {
            "keyword_searched": keyword,
            "omop_table": omop_table,
            "omop_field": omop_field,
            "total_found": len(concepts),
            "candidates_returned": len(candidates),
            "selection_guidance": (
                "Select the most appropriate concept based on clinical context. "
                "Access omop://preferred_vocabularies for vocabulary preferences. "
                "Generally prefer Standard + Valid concepts from recommended vocabularies, "
                "but context may require different choices (e.g., research needs, "
                "specific vocabulary requirements, or non-standard mappings)."
            ),
        },
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
            "reason",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            keyword = row.get("keyword", "")
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
