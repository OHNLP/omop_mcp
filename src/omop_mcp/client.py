"""
OMOPHub API Client — Reusable client for the OMOPHub Vocabulary API.
"""

import asyncio
from typing import Any

import httpx


class OMOPHubClient:
    """Client for OMOPHub API"""

    def __init__(self, api_key: str, base_url: str = "https://api.omophub.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> httpx.Response:
        """Make an API request with automatic retry on 429 rate limits."""
        response = None
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await getattr(client, method)(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=self.headers,
                )
                if response.status_code == 429:
                    wait = float(response.headers.get("Retry-After", 2**attempt))
                    await asyncio.sleep(min(wait, 10))
                    continue
                return response
        return response

    async def search_concepts(
        self,
        query: str,
        search_type: str = "keyword",
        vocabulary_ids: list[str] | None = None,
        domain_id: str | None = None,
        standard_concept: bool | None = None,
        page_size: int = 20,
        page: int = 1,
    ) -> dict[str, Any]:
        """
        Search concepts via OMOPHub API.
        For concept_id search, uses the direct concept endpoint.
        For keyword/code/name, uses the search endpoint.
        """
        if search_type == "concept_id":
            return await self._search_by_concept_id(query)

        params: dict[str, Any] = {"query": query, "page_size": page_size, "page": page}

        if vocabulary_ids:
            params["vocabulary_ids"] = ",".join(vocabulary_ids)
        if domain_id:
            params["domain_id"] = domain_id
        if standard_concept is not None:
            params["standard_concept"] = "S" if standard_concept else "C"

        response = await self._request("get", "/search/concepts", params=params)
        response.raise_for_status()
        raw = response.json()

        return {
            "results": raw.get("data", []),
            "total": raw.get("meta", {}).get("pagination", {}).get("total_items", 0),
            "page": raw.get("meta", {}).get("pagination", {}).get("page", page),
            "page_size": raw.get("meta", {})
            .get("pagination", {})
            .get("page_size", page_size),
        }

    async def suggest_concepts(
        self,
        query: str,
        page_size: int = 20,
        page: int = 1,
        vocabulary_ids: list[str] | None = None,
        domain_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get concept suggestions from OMOPHub suggest_concepts API.
        Optimized for partial matches and finding standard concepts/ingredients
        (Athena-style search).
        """
        params: dict[str, Any] = {"query": query, "page_size": page_size, "page": page}

        if vocabulary_ids:
            params["vocabulary_ids"] = ",".join(vocabulary_ids)
        if domain_id:
            params["domain_id"] = domain_id

        response = await self._request(
            "get",
            "/concepts/suggest",
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        raw = response.json()
        data = raw.get("data", [])
        meta = raw.get("meta", {})
        pagination = meta.get("pagination", {})

        return {
            "results": data if isinstance(data, list) else [],
            "total": pagination.get(
                "total_items", len(data) if isinstance(data, list) else 0
            ),
            "page": pagination.get("page", page),
            "page_size": pagination.get("page_size", page_size),
        }

    async def _search_by_concept_id(self, concept_id_str: str) -> dict[str, Any]:
        """Look up a concept by its concept_id using the direct endpoint."""
        try:
            concept_id = int(concept_id_str.strip())
        except ValueError:
            return {"results": [], "total": 0, "page": 1, "page_size": 1}

        response = await self._request("get", f"/concepts/{concept_id}")
        if response.status_code == 404:
            return {"results": [], "total": 0, "page": 1, "page_size": 1}
        response.raise_for_status()
        raw = response.json()
        concept = raw.get("data", {})
        if concept:
            return {"results": [concept], "total": 1, "page": 1, "page_size": 1}
        return {"results": [], "total": 0, "page": 1, "page_size": 1}

    async def get_concept(self, concept_id: int) -> dict[str, Any]:
        response = await self._request("get", f"/concepts/{concept_id}")
        response.raise_for_status()
        raw = response.json()
        return raw.get("data", {})

    async def get_vocabularies(self) -> list[dict[str, Any]]:
        response = await self._request(
            "get",
            "/vocabularies",
            params={"page_size": 1000},
        )
        response.raise_for_status()
        raw = response.json()
        data = raw.get("data", [])
        if isinstance(data, dict):
            if "vocabularies" in data:
                return data["vocabularies"]
        return data

    async def get_domains(self) -> list[dict[str, Any]]:
        response = await self._request(
            "get",
            "/domains",
            params={"page_size": 1000},
        )
        response.raise_for_status()
        raw = response.json()
        data = raw.get("data", [])
        if isinstance(data, dict):
            if "domains" in data:
                return data["domains"]
        return data
