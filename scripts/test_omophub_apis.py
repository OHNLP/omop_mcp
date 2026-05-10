#!/usr/bin/env python3
"""
Smoke test for OMOPHub API — run with: source .venv/bin/activate && python scripts/test_omophub_apis.py
"""
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

OMOPHUB_API_KEY = os.getenv("OMOPHUB_API_KEY")
OMOPHUB_BASE_URL = os.getenv("OMOPHUB_BASE_URL", "https://api.omophub.com/v1")

if not OMOPHUB_API_KEY:
    raise ValueError("OMOPHUB_API_KEY not found in .env file")

HEADERS = {
    "Authorization": f"Bearer {OMOPHUB_API_KEY}",
    "Content-Type": "application/json",
}


def test(name: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, requests.Response):
            status = "OK" if result.status_code == 200 else f"HTTP {result.status_code}"
        else:
            status = "OK"
        print(f"  [{status}] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


# ========== Core endpoints ==========

def search(query: str, page_size: int = 5):
    r = requests.get(f"{OMOPHUB_BASE_URL}/search/concepts", headers=HEADERS, params={"query": query, "page_size": page_size}, timeout=10)
    r.raise_for_status()
    return r


def get_concept(concept_id: int):
    r = requests.get(f"{OMOPHUB_BASE_URL}/concepts/{concept_id}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r


def get_domains():
    r = requests.get(f"{OMOPHUB_BASE_URL}/domains", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r


def list_vocabularies():
    r = requests.get(f"{OMOPHUB_BASE_URL}/vocabularies", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r


def bulk_search(queries: list[str]):
    r = requests.post(f"{OMOPHUB_BASE_URL}/search/bulk", headers=HEADERS, json={"queries": queries}, timeout=10)
    r.raise_for_status()
    return r


def get_relationships(concept_id: int):
    r = requests.get(f"{OMOPHUB_BASE_URL}/concepts/{concept_id}/relationships", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r


def main():
    print(f"OMOPHub API smoke test — {datetime.now().isoformat()}")
    print(f"Base URL: {OMOPHUB_BASE_URL}\n")

    passed = 0
    total = 0

    # Get a concept_id to use in subsequent tests
    test_concept_id = 201826
    try:
        r = requests.get(f"{OMOPHUB_BASE_URL}/search/concepts", headers=HEADERS, params={"query": "diabetes", "page_size": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("data") and len(data["data"]) > 0:
            test_concept_id = data["data"][0].get("concept_id", 201826)
    except Exception:
        pass

    print(f"Using test concept_id: {test_concept_id}\n")

    total += 1
    if test("Search concepts (diabetes)", search, "diabetes"):
        passed += 1

    total += 1
    if test("Get concept by ID", get_concept, test_concept_id):
        passed += 1

    total += 1
    if test("Bulk search", bulk_search, ["diabetes", "hypertension"]):
        passed += 1

    total += 1
    if test("Get concept relationships", get_relationships, test_concept_id):
        passed += 1

    total += 1
    if test("List domains", get_domains):
        passed += 1

    total += 1
    if test("List vocabularies", list_vocabularies):
        passed += 1

    print(f"\n{passed}/{total} passed")

    results = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "test_concept_id": test_concept_id,
    }
    output_file = Path(__file__).parent.parent / "omophub_smoke_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
