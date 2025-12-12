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
        Dict containing candidate concepts or error information.
    """
    logging.info(
        f"find_omop_concept called with keyword='{keyword}', omop_table='{omop_table}', omop_field='{omop_field}'"
    )

    # Create a new session for each request
    async with aiohttp.ClientSession() as session:
        url = "https://athena.ohdsi.org/api/v1/concepts"
        params = {"query": keyword}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://athena.ohdsi.org/search-terms",
            "Origin": "https://athena.ohdsi.org",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }

        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        except aiohttp.ClientError as e:
            return {
                "error": f"Failed to query Athena: {str(e)}",
            }

        logging.debug(f"Athena response: {data}")
        concepts = []
        if isinstance(data, dict) and "content" in data:
            concepts = data["content"]
        elif isinstance(data, list):
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

        # Return multiple candidates with all their metadata for LLM to evaluate
        candidates = []
        for i, c in enumerate(concepts[:max_results]):
            candidate = {
                "concept_id": c.get("id", ""),
                "code": c.get("code", ""),
                "name": c.get("name", ""),
                "class": c.get("className", ""),
                "concept": c.get("standardConcept", ""),
                "validity": c.get("invalidReason", c.get("validity", "")),
                "domain": c.get("domain", c.get("domainId", "")),
                "vocab": c.get("vocabulary", c.get("vocabularyId", "")),
                "url": f"https://athena.ohdsi.org/search-terms/terms/{c.get('id', '')}",
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
