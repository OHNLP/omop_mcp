import re

import aiohttp
import requests

# Shared constants
ATHENA_SEARCH_URL = "https://athena.ohdsi.org/search-terms"
ATHENA_API_URL = "https://athena.ohdsi.org/api/v1/concepts"

# Shared headers
INITIAL_PAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://athena.ohdsi.org/search-terms",
    "Origin": "https://athena.ohdsi.org",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
}


def _extract_concepts_from_response(data):
    """
    Extract concepts list from Athena API response.
    Handles various response formats.
    """
    if isinstance(data, dict) and "content" in data:
        return data["content"]
    elif isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for key in ("content", "results", "items", "concepts"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def search_athena_concept(keyword: str):
    """
    Search for OMOP concepts using the Athena web interface.
    This function scrapes the search results from the Athena website.
    """
    session = requests.Session()
    try:
        session.get(ATHENA_SEARCH_URL, headers=INITIAL_PAGE_HEADERS, timeout=10)
    except Exception:
        pass

    params = {"query": keyword}

    try:
        response = session.get(
            ATHENA_API_URL, params=params, headers=API_HEADERS, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return _extract_concepts_from_response(data)
    except requests.exceptions.RequestException as e:
        print(f"Error searching Athena: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


async def search_athena_concept_async(keyword: str):
    """
    Async version of search_athena_concept using aiohttp.

    Raises RuntimeError for API failures (connection/HTTP errors).
    Returns empty list for successful API calls with no results.
    """
    async with aiohttp.ClientSession() as session:
        # First, visit the search page to establish a session and get cookies
        try:
            async with session.get(
                ATHENA_SEARCH_URL,
                headers=INITIAL_PAGE_HEADERS,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as _:
                pass
        except aiohttp.ClientError as e:
            raise RuntimeError(
                f"Failed to establish session with Athena API: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error establishing Athena session: {e}"
            ) from e

        params = {"query": keyword}

        try:
            async with session.get(
                ATHENA_API_URL,
                params=params,
                headers=API_HEADERS,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return _extract_concepts_from_response(data)
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Athena API request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Athena API: {e}") from e


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
