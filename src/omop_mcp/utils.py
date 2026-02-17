import asyncio
import logging
import os
import re

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
    provider: str = "azure_openai",
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    temperature: float = 0,
    **kwargs,
):
    """
    Centralized factory for creating LLM instances.
    """
    # Default models for each provider
    provider_defaults = {
        "azure_openai": "gpt-4o",
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20240620",
        "gemini": "gemini-2.0-flash",
        "ollama": "llama3",
        "openrouter": "google/gemma-3-27b-it:free",
        "groq": "llama-3.3-70b-versatile",
        "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    }

    # Use provider-specific model if provided,
    # then fallback to global MODEL_NAME (but only if it seems appropriate for the provider)
    # else use the provider default.
    effective_model = model
    if not effective_model:
        global_model = os.getenv("MODEL_NAME")
        if global_model:
            # Heuristic: if global model is 'gpt-...' and we are not an OpenAI/Azure provider,
            # use the provider default instead.
            is_openai_model = "gpt" in global_model.lower()
            non_openai_provider = provider not in ["openai", "azure_openai"]
            if non_openai_provider and is_openai_model:
                effective_model = provider_defaults.get(provider)
            else:
                effective_model = global_model
        else:
            effective_model = provider_defaults.get(provider)

    # Azure OpenAI
    if provider == "azure_openai":
        if not AzureChatOpenAI:
            raise ImportError("langchain-openai is not installed.")
        return AzureChatOpenAI(
            azure_deployment=effective_model,
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
            model=effective_model,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            temperature=temperature,
            **kwargs,
        )

    # Anthropic
    elif provider == "anthropic":
        if not ChatAnthropic:
            raise ImportError("langchain-anthropic is not installed.")
        return ChatAnthropic(
            model=effective_model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=temperature,
            **kwargs,
        )

    # Gemini
    elif provider == "gemini":
        if not ChatGoogleGenerativeAI:
            raise ImportError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model=effective_model,
            api_key=api_key or os.getenv("GOOGLE_API_KEY", ""),
            temperature=temperature,
            **kwargs,
        )

    # Ollama
    elif provider == "ollama":
        if not ChatOpenAI:
            raise ImportError("langchain-openai is not installed.")
        base = (endpoint or "http://localhost:11434").rstrip("/")
        return ChatOpenAI(
            model=effective_model,
            base_url=f"{base}/v1",
            api_key="ollama",
            temperature=temperature,
            **kwargs,
        )

    # OpenRouter
    elif provider == "openrouter":
        return _get_openrouter(effective_model, api_key, temperature, **kwargs)

    # Groq
    elif provider == "groq":
        return _get_groq(effective_model, api_key, temperature, **kwargs)

    # HuggingFace
    elif provider == "huggingface":
        return _get_huggingface(effective_model, api_key, temperature, **kwargs)

    else:
        from omop_mcp.config import LLM_PROVIDERS

        raise ValueError(
            f"Unsupported llm_provider: '{provider}'. "
            f"Use one of: {', '.join(LLM_PROVIDERS)}."
        )


def _get_openrouter(
    model: str | None, api_key: str | None, temperature: float, **kwargs
):
    if not ChatOpenAI:
        raise ImportError("langchain-openai is not installed.")

    return ChatOpenAI(
        model=model or "google/gemma-3-27b-it:free",
        api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=kwargs.pop("max_tokens", 4000),
        **kwargs,
    )


def _get_groq(model: str | None, api_key: str | None, temperature: float, **kwargs):
    if not ChatOpenAI:
        raise ImportError("langchain-openai is not installed.")

    return ChatOpenAI(
        model=model,
        api_key=api_key or os.getenv("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
        temperature=temperature,
        max_tokens=kwargs.pop("max_tokens", 4000),
        **kwargs,
    )


def _get_huggingface(
    model: str | None, api_key: str | None, temperature: float, **kwargs
):
    # Try importing from new package first, then community
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except ImportError:
        try:
            from langchain_community.llms import HuggingFaceEndpoint

            ChatHuggingFace = None
        except ImportError:
            raise ImportError(
                "langchain-huggingface or langchain-community is not installed."
            )

    llm = HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
        temperature=temperature,
        **kwargs,
    )

    if ChatHuggingFace:
        return ChatHuggingFace(llm=llm)
    return llm


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
    markdown_link_pattern = r"\[([^\]]+)\]\((https://athena\.ohdsi\.org/[^)]+)\)"

    def replace_markdown_link(match):
        url = match.group(2)
        return url

    return re.sub(markdown_link_pattern, replace_markdown_link, response)
