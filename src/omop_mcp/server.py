import json

import requests
from mcp.server.fastmcp import FastMCP

# Load OMOP CDM table/field mapping from JSON file
with open("src/omop_mcp/data/omop_concept_id_fields.json", "r") as f:
    OMOP_CDM = json.load(f)

mcp = FastMCP("omop_concept_mapper")


@mcp.tool()
def find_omop_concept(keyword: str, omop_table: str, omop_field: str) -> dict:
    """Find the best-matching OMOP concept for a given keyword, table, and field."""
    # Validate OMOP table and field
    if omop_table not in OMOP_CDM:
        return {"error": (f"OMOP table '{omop_table}' not found in OMOP CDM.")}
    if omop_field not in OMOP_CDM[omop_table]:
        return {
            "error": (
                f"Field '{omop_field}' not found in OMOP table " f"'{omop_table}'."
            )
        }

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
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"error": (f"Failed to query Athena: {e}")}

    concepts = []
    if isinstance(data, list):
        concepts = data
    elif isinstance(data, dict):
        for key in ("content", "results", "items", "concepts"):
            if key in data and isinstance(data[key], list):
                concepts = data[key]
                break

    if not concepts:
        return {"error": ("No results found or unexpected response structure.")}

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
    }


if __name__ == "__main__":
    mcp.serve()
