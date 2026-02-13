import asyncio
import logging
import os
import re
from typing import Literal

import httpx
from dotenv import load_dotenv

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
except ImportError:
    AzureChatOpenAI = None
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

load_dotenv()

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.omophub.com/v1")


def get_api_headers() -> dict:
    """
    Get API headers.
    """
    api_key = os.getenv("OMOPHUB_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_llm(
    provider: Literal[
        "azure_openai", "openai", "anthropic", "gemini", "ollama"
    ] = "azure_openai",
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    temperature: float = 0,
    **kwargs,
):
    """
    Centralized factory for creating LLM instances.
    """
    # Azure OpenAI
    if provider == "azure_openai":
        if not AzureChatOpenAI:
            raise ImportError("langchain-openai is not installed.")
        return AzureChatOpenAI(
            azure_deployment=model or os.getenv("MODEL_NAME", "gpt-4o"),
            azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            temperature=temperature,
            **kwargs,
        )

    # OpenAI
    elif provider == "openai":
        if not ChatOpenAI:
            raise ImportError("langchain-openai is not installed.")
        return ChatOpenAI(
            model=model or os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            temperature=temperature,
            **kwargs,
        )

    # Anthropic
    elif provider == "anthropic":
        if not ChatAnthropic:
            raise ImportError("langchain-anthropic is not installed.")
        return ChatAnthropic(
            model=model or os.getenv("MODEL_NAME", "claude-sonnet-4-20250514"),
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=temperature,
            **kwargs,
        )

    # Gemini
    elif provider == "gemini":
        if not ChatGoogleGenerativeAI:
            raise ImportError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("MODEL_NAME", "gemini-2.5-flash"),
            api_key=api_key or os.getenv("GOOGLE_API_KEY", ""),
            temperature=temperature,
            **kwargs,
        )

    # Ollama
    elif provider == "ollama":
        # Ollama typically uses ChatOpenAI client with a different base_url
        if not ChatOpenAI:
            raise ImportError("langchain-openai is not installed.")
        base = (endpoint or "http://localhost:11434").rstrip("/")
        return ChatOpenAI(
            model=model or "llama3",
            base_url=f"{base}/v1",
            api_key="ollama",  # dummy key required
            temperature=temperature,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported llm_provider: '{provider}'. "
            "Use one of: azure_openai, openai, anthropic, gemini, ollama."
        )


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------


async def _suggest_concepts_async(
    client: httpx.AsyncClient, keyword: str, limit: int = 10
) -> list:
    """suggest_concepts endpoint (capped at ~10 by server)."""
    url = f"{API_BASE_URL}/concepts/suggest"
    params = {"query": keyword, "limit": limit}
    response = await client.get(
        url,
        headers=get_api_headers(),
        params=params,
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json().get("data", [])


async def _basic_search_async(
    client: httpx.AsyncClient, keyword: str, page_size: int = 20
) -> list:
    """basic_search endpoint (supports larger page_size)."""
    url = f"{API_BASE_URL}/search/concepts"
    params = {"query": keyword, "page_size": page_size}
    response = await client.get(
        url,
        headers=get_api_headers(),
        params=params,
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json().get("data", [])


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
        response = httpx.get(
            url, headers=get_api_headers(), params=params, timeout=15.0
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except httpx.RequestError as e:
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

    async with httpx.AsyncClient(timeout=15.0) as client:
        suggest_task = _suggest_concepts_async(client, keyword, limit=10)
        basic_task = _basic_search_async(client, keyword, page_size=basic_page_size)

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
        response = httpx.get(url, headers=get_api_headers(), timeout=10.0)
        response.raise_for_status()
        return response.json().get("data")
    except httpx.RequestError:
        return None


def concept_id_exists(concept_id: int | str) -> bool:
    """Check if a concept exists."""
    return get_concept_by_id(concept_id) is not None


def get_concept_name(concept_id: int | str) -> str | None:
    """Get concept name by ID."""
    concept = get_concept_by_id(concept_id)
    return concept.get("concept_name") if concept else None


def parse_agent_response(response):
    _FIELDS = [
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
    _ALIASES = {
        "concept_id": ["CONCEPT_ID", "Concept ID"],
        "code": ["CODE", "Code"],
        "name": ["NAME", "Name"],
        "class": ["CLASS", "Class"],
        "concept": ["CONCEPT", "Concept"],
        "validity": ["VALIDITY", "Validity"],
        "domain": ["DOMAIN", "Domain"],
        "vocab": ["VOCAB", "Vocabulary"],
        "url": ["URL"],
        "reason": ["REASON", "Reason"],
    }

    def _extract(field):
        aliases = _ALIASES.get(field, [field.upper()])
        for alias in aliases:
            # Multi-line capture for the last field (reason)
            if field == "reason":
                pat = rf"(?:\*\*)?{re.escape(alias)}(?:\*\*)?:\s*([\s\S]+)"
            else:
                pat = rf"(?:\*\*)?{re.escape(alias)}(?:\*\*)?:\s*([^\n]+)"
            match = re.search(pat, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # Fallback: bare Athena URL for the url field
        if field == "url":
            match = re.search(r"(https://athena\.ohdsi\.org[^\s\)]+)", response)
            if match:
                return match.group(1).strip()
        return ""

    return {f: _extract(f) for f in _FIELDS}


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
