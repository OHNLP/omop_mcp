import re

import requests


def search_athena_concept(keyword: str):
    """
    Search for OMOP concepts using the Athena web interface.
    This function scrapes the search results from the Athena website.
    """
    url = "https://athena.ohdsi.org/api/v1/concepts"
    params = {"query": keyword}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://athena.ohdsi.org/search-terms",
        "Origin": "https://athena.ohdsi.org",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract concepts from the response
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
        else:
            concepts = []

        return concepts

    except requests.exceptions.RequestException as e:
        print(f"Error searching Athena: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def concept_id_exists_in_athena(concept_id: str) -> bool:
    """Check if a concept exists in Athena."""
    results = search_athena_concept(concept_id)
    for concept in results:
        if str(concept.get("id")) == str(concept_id):
            return True
    return False


def get_concept_name_from_athena(concept_id: str) -> str | None:
    """Get the matching concept name from concept ID."""
    results = search_athena_concept(concept_id)
    for concept in results:
        if str(concept.get("id")) == str(concept_id):
            return concept.get("name")
    return None


def parse_agent_response(response):
    def extract(patterns):
        for pat in patterns:
            match = re.search(pat, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    return {
        "concept_id": extract(
            [
                r"\*\*CONCEPT_ID\*\*:\s*([^\n]+)",
                r"\*\*Concept ID\*\*:\s*([^\n]+)",
                r"CONCEPT_ID:\s*([^\n]+)",
                r"Concept ID:\s*([^\n]+)",
            ]
        ),
        "code": extract(
            [
                r"\*\*CODE\*\*:\s*([^\n]+)",
                r"\*\*Code\*\*:\s*([^\n]+)",
                r"CODE:\s*([^\n]+)",
                r"Code:\s*([^\n]+)",
            ]
        ),
        "name": extract(
            [
                r"\*\*NAME\*\*:\s*([^\n]+)",
                r"\*\*Name\*\*:\s*([^\n]+)",
                r"NAME:\s*([^\n]+)",
                r"Name:\s*([^\n]+)",
            ]
        ),
        "class": extract(
            [
                r"\*\*CLASS\*\*:\s*([^\n]+)",
                r"\*\*Class\*\*:\s*([^\n]+)",
                r"CLASS:\s*([^\n]+)",
                r"Class:\s*([^\n]+)",
            ]
        ),
        "concept": extract(
            [
                r"\*\*CONCEPT\*\*:\s*([^\n]+)",
                r"\*\*Concept\*\*:\s*([^\n]+)",
                r"CONCEPT:\s*([^\n]+)",
                r"Concept:\s*([^\n]+)",
            ]
        ),
        "validity": extract(
            [
                r"\*\*VALIDITY\*\*:\s*([^\n]+)",
                r"\*\*Validity\*\*:\s*([^\n]+)",
                r"VALIDITY:\s*([^\n]+)",
                r"Validity:\s*([^\n]+)",
            ]
        ),
        "domain": extract(
            [
                r"\*\*DOMAIN\*\*:\s*([^\n]+)",
                r"\*\*Domain\*\*:\s*([^\n]+)",
                r"DOMAIN:\s*([^\n]+)",
                r"Domain:\s*([^\n]+)",
            ]
        ),
        "vocab": extract(
            [
                r"\*\*VOCAB\*\*:\s*([^\n]+)",
                r"\*\*Vocabulary\*\*:\s*([^\n]+)",
                r"VOCAB:\s*([^\n]+)",
                r"Vocabulary:\s*([^\n]+)",
            ]
        ),
        "url": extract(
            [
                r"\*\*URL\*\*:\s*([^\n]+)",
                r"(https://athena\.ohdsi\.org[^\s\)]+)",
                r"URL:\s*([^\n]+)",
            ]
        ),
        "reason": extract(
            [
                r"\*\*REASON\*\*:\s*([^\n]+)",
                r"\*\*Reason\*\*:\s*([^\n]+)",
                r"REASON:\s*([^\n]+)",
                r"Reason:\s*([^\n]+)",
            ]
        ),
    }


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
