import asyncio
import logging
import os
import re

import aiohttp
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.omophub.com/v1")
API_KEY = os.getenv("OMOPHUB_API_KEY", "")

API_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------


async def _suggest_concepts_async(
    session: aiohttp.ClientSession, keyword: str, limit: int = 10
) -> list:
    """suggest_concepts endpoint (capped at ~10 by server)."""
    url = f"{API_BASE_URL}/concepts/suggest"
    params = {"query": keyword, "limit": limit}
    async with session.get(
        url,
        headers=API_HEADERS,
        params=params,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as resp:
        resp.raise_for_status()
        return (await resp.json()).get("data", [])


async def _basic_search_async(
    session: aiohttp.ClientSession, keyword: str, page_size: int = 20
) -> list:
    """basic_search endpoint (supports larger page_size)."""
    url = f"{API_BASE_URL}/search/concepts"
    params = {"query": keyword, "page_size": page_size}
    async with session.get(
        url,
        headers=API_HEADERS,
        params=params,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as resp:
        resp.raise_for_status()
        return (await resp.json()).get("data", [])


def _merge_and_dedup(primary: list, secondary: list) -> list:
    """Merge two concept lists, deduplicating by concept_id. Primary results come first."""
    seen = set()
    merged = []
    for c in primary:
        cid = c.get("concept_id")
        if cid not in seen:
            seen.add(cid)
            merged.append(c)
    for c in secondary:
        cid = c.get("concept_id")
        if cid not in seen:
            seen.add(cid)
            merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------


def search_concepts(keyword: str, max_results: int = 20) -> list:
    """Search OMOP concepts via suggest_concepts API (sync)."""
    url = f"{API_BASE_URL}/concepts/suggest"
    params = {"query": keyword, "limit": max_results}

    try:
        response = requests.get(url, headers=API_HEADERS, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Suggest API error: {e}")
        return []


async def search_concepts_async(keyword: str, max_results: int = 20) -> list:
    """Combined search: suggest_concepts + basic_search in parallel, merged & deduped.

    suggest_concepts returns ~10 high-quality standard/ingredient-level concepts.
    basic_search returns up to 20 broader product-level concepts.
    Results are merged with suggest_concepts first (higher relevance).

    Raises RuntimeError if both APIs fail.
    """
    basic_page_size = max(max_results - 10, 10)

    async with aiohttp.ClientSession() as session:
        suggest_task = _suggest_concepts_async(session, keyword, limit=10)
        basic_task = _basic_search_async(session, keyword, page_size=basic_page_size)

        results = await asyncio.gather(suggest_task, basic_task, return_exceptions=True)
        suggest_results, basic_results = results

        if isinstance(suggest_results, Exception) and isinstance(
            basic_results, Exception
        ):
            raise RuntimeError(
                f"Both APIs failed — suggest: {suggest_results}, basic: {basic_results}"
            ) from suggest_results

        if isinstance(suggest_results, Exception):
            logger.warning(
                f"suggest_concepts failed, using basic_search only: {suggest_results}"
            )
            suggest_results = []
        if isinstance(basic_results, Exception):
            logger.warning(
                f"basic_search failed, using suggest_concepts only: {basic_results}"
            )
            basic_results = []

        merged = _merge_and_dedup(suggest_results, basic_results)
        return merged[:max_results]


def get_concept_by_id(concept_id: int | str) -> dict | None:
    """Get a single concept by ID."""
    url = f"{API_BASE_URL}/concepts/{concept_id}"
    try:
        response = requests.get(url, headers=API_HEADERS, timeout=10)
        response.raise_for_status()
        return response.json().get("data")
    except requests.exceptions.RequestException:
        return None


def concept_id_exists(concept_id: int | str) -> bool:
    """Check if a concept exists."""
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
