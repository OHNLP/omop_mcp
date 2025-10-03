import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
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


class VocabDBService:
    """Service for querying local OMOP vocabulary database."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use relative path from project root
            project_root = Path(__file__).parent.parent.parent
            self.db_path = project_root / "data" / "vocab" / "omop_vocab.db"
        else:
            self.db_path = Path(db_path)

        self.is_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if local database is available."""
        try:
            if not self.db_path.exists():
                return False
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("SELECT COUNT(*) FROM concept LIMIT 1")
            conn.close()
            return True
        except Exception:
            return False

    def search_concepts(
        self, keyword: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Search for concepts in local database."""
        if not self.is_available:
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))

            # Search in both concept_name and concept_synonym
            query = """
            SELECT DISTINCT 
                c.concept_id,
                c.concept_name,
                c.concept_code,
                c.vocabulary_id,
                c.domain_id,
                c.concept_class_id,
                c.standard_concept,
                c.invalid_reason
            FROM concept c
            LEFT JOIN concept_synonym cs ON c.concept_id = cs.concept_id
            WHERE 
                c.concept_name LIKE ? 
                OR cs.concept_synonym_name LIKE ?
                OR c.concept_code LIKE ?
            ORDER BY 
                CASE WHEN c.standard_concept = 'S' THEN 1 ELSE 2 END,
                c.concept_name
            LIMIT ?
            """

            search_term = f"%{keyword}%"
            results = pd.read_sql(
                query, conn, params=[search_term, search_term, search_term, max_results]
            )
            conn.close()

            return results.to_dict("records")

        except Exception as e:
            print(f"Error searching local database: {e}")
            return []

    def get_database_status(self) -> dict[str, Any]:
        """Get status of local vocabulary database."""
        if not self.is_available:
            return {
                "status": "unavailable",
                "error": "Local vocabulary database not found or corrupted",
                "database_path": str(self.db_path),
            }

        try:
            conn = sqlite3.connect(str(self.db_path))

            # Get basic stats
            total_concepts = pd.read_sql("SELECT COUNT(*) as count FROM concept", conn)[
                "count"
            ].iloc[0]

            # Get vocabulary counts
            vocab_counts = pd.read_sql(
                """
                SELECT vocabulary_id, COUNT(*) as count 
                FROM concept 
                GROUP BY vocabulary_id 
                ORDER BY count DESC 
                LIMIT 10
            """,
                conn,
            )

            # Get domain counts
            domain_counts = pd.read_sql(
                """
                SELECT domain_id, COUNT(*) as count 
                FROM concept 
                GROUP BY domain_id 
                ORDER BY count DESC
            """,
                conn,
            )

            conn.close()

            return {
                "status": "available",
                "database_path": str(self.db_path),
                "total_concepts": int(total_concepts),
                "top_vocabularies": vocab_counts.to_dict("records"),
                "domains": domain_counts.to_dict("records"),
                "last_checked": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "database_path": str(self.db_path),
            }

    def search_by_vocabulary(
        self, keyword: str, vocabulary_id: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Search for concepts in a specific vocabulary."""
        if not self.is_available:
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))

            query = """
            SELECT DISTINCT 
                c.concept_id,
                c.concept_name,
                c.concept_code,
                c.vocabulary_id,
                c.domain_id,
                c.concept_class_id,
                c.standard_concept,
                c.invalid_reason
            FROM concept c
            LEFT JOIN concept_synonym cs ON c.concept_id = cs.concept_id
            WHERE 
                c.vocabulary_id = ?
                AND (c.concept_name LIKE ? 
                     OR cs.concept_synonym_name LIKE ?
                     OR c.concept_code LIKE ?)
            ORDER BY 
                CASE WHEN c.standard_concept = 'S' THEN 1 ELSE 2 END,
                c.concept_name
            LIMIT ?
            """

            search_term = f"%{keyword}%"
            results = pd.read_sql(
                query,
                conn,
                params=[
                    vocabulary_id,
                    search_term,
                    search_term,
                    search_term,
                    max_results,
                ],
            )
            conn.close()

            return results.to_dict("records")

        except Exception as e:
            print(f"Error searching local database: {e}")
            return []

    def get_concept_by_id(self, concept_id: int) -> dict[str, Any] | None:
        """Get a specific concept by its ID."""
        if not self.is_available:
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))

            query = """
            SELECT 
                c.concept_id,
                c.concept_name,
                c.concept_code,
                c.vocabulary_id,
                c.domain_id,
                c.concept_class_id,
                c.standard_concept,
                c.invalid_reason
            FROM concept c
            WHERE c.concept_id = ?
            """

            result = pd.read_sql(query, conn, params=[concept_id])
            conn.close()

            if result.empty:
                return None

            return result.iloc[0].to_dict()

        except Exception as e:
            print(f"Error getting concept by ID: {e}")
            return None
