import requests


def search_athena_concept(keyword: str):
    url = "https://athena.ohdsi.org/api/v1/concepts"
    params = {"query": keyword}
    headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    # The response may be a list or a dict with a list under a key
    if isinstance(data, list):
        concepts = data
    elif isinstance(data, dict):
        for key in ("content", "results", "items", "concepts"):
            if key in data and isinstance(data[key], list):
                concepts = data[key]
                break
        else:
            concepts = []
    else:
        concepts = []
    return concepts


def concept_exists_in_athena(concept_id: str) -> bool:
    """Check if a concept exists in Athena."""
    results = search_athena_concept(concept_id)
    for concept in results:
        if str(concept.get("id")) == str(concept_id):
            return True
    return False
