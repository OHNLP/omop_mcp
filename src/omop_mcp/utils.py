import re

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
    """
    Parse the agent response to extract structured information
    """
    # Extract concept ID
    id_match = re.search(r"\*\*CONCEPT_ID\*\*:\s*(\d+)", response)
    concept_id = id_match.group(1) if id_match else ""

    # Extract code
    code_match = re.search(r"\*\*CODE\*\*:\s*(.+?)(?:\n|$)", response)
    code = code_match.group(1).strip() if code_match else ""

    # Extract name
    name_match = re.search(r"\*\*NAME\*\*:\s*(.+?)(?:\n|$)", response)
    name = name_match.group(1).strip() if name_match else ""

    # Extract class
    class_match = re.search(r"\*\*CLASS\*\*:\s*(.+?)(?:\n|$)", response)
    class_val = class_match.group(1).strip() if class_match else ""

    # Extract concept
    concept_match = re.search(r"\*\*CONCEPT\*\*:\s*(.+?)(?:\n|$)", response)
    concept = concept_match.group(1).strip() if concept_match else ""

    # Extract validity
    validity_match = re.search(r"\*\*VALIDITY\*\*:\s*(.+?)(?:\n|$)", response)
    validity = validity_match.group(1).strip() if validity_match else ""

    # Extract domain
    domain_match = re.search(r"\*\*DOMAIN\*\*:\s*(.+?)(?:\n|$)", response)
    domain = domain_match.group(1).strip() if domain_match else ""

    # Extract vocabulary
    vocab_match = re.search(r"\*\*VOCAB\*\*:\s*(.+?)(?:\n|$)", response)
    vocab = vocab_match.group(1).strip() if vocab_match else ""

    # Extract URL
    url_match = re.search(r"\*\*URL\*\*:\s*(.+?)(?:\n|$)", response)
    url = url_match.group(1).strip() if url_match else ""

    reason_match = re.search(r"\*\*REASON\*\*:\s*(.+?)(?:\n|$)", response)
    reason = reason_match.group(1).strip() if reason_match else ""

    return {
        "concept_id": concept_id,
        "code": code,
        "name": name,
        "class": class_val,
        "concept": concept,
        "validity": validity,
        "domain": domain,
        "vocab": vocab,
        "url": url,
        "reason": reason,
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
