import os
import re

import aiohttp
import requests
from dotenv import load_dotenv

load_dotenv()

OMOPHUB_BASE_URL = os.getenv("OMOPHUB_BASE_URL", "https://api.omophub.com/v1")
OMOPHUB_API_KEY = os.getenv("OMOPHUB_API_KEY", "")

OMOPHUB_HEADERS = {
    "Authorization": f"Bearer {OMOPHUB_API_KEY}",
    "Content-Type": "application/json",
}


def search_omophub_concepts(keyword: str, max_results: int = 20) -> list:
    """Search OMOP concepts via OMOPHub suggest_concepts API (sync)."""
    url = f"{OMOPHUB_BASE_URL}/concepts/suggest"
    params = {"query": keyword, "limit": max_results}

    try:
        response = requests.get(url, headers=OMOPHUB_HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"OMOPHub API error: {e}")
        return []


async def search_omophub_concepts_async(keyword: str, max_results: int = 20) -> list:
    """Search OMOP concepts via OMOPHub suggest_concepts API (async).

    Raises RuntimeError for API failures.
    Returns empty list for successful calls with no results.
    """
    url = f"{OMOPHUB_BASE_URL}/concepts/suggest"
    params = {"query": keyword, "limit": max_results}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                url,
                headers=OMOPHUB_HEADERS,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("data", [])
        except aiohttp.ClientError as e:
            raise RuntimeError(f"OMOPHub API request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling OMOPHub API: {e}") from e


def get_concept_by_id(concept_id: int | str) -> dict | None:
    """Get a single concept by ID via OMOPHub API."""
    url = f"{OMOPHUB_BASE_URL}/concepts/{concept_id}"
    try:
        response = requests.get(url, headers=OMOPHUB_HEADERS, timeout=10)
        response.raise_for_status()
        return response.json().get("data")
    except requests.exceptions.RequestException:
        return None


def concept_id_exists(concept_id: int | str) -> bool:
    """Check if a concept exists in OMOPHub."""
    return get_concept_by_id(concept_id) is not None


def get_concept_name(concept_id: int | str) -> str | None:
    """Get concept name by ID."""
    concept = get_concept_by_id(concept_id)
    return concept.get("concept_name") if concept else None


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
