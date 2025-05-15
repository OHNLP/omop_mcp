# OMOP MCP Server

Model Context Protocol (MCP) server for mapping clinical terminology to Observational Medical Outcomes Partnership (OMOP) concepts using Large Language Models (LLMs).

## Installation

### Configuration for Claude Desktop

Add the following configuration to your `claude_desktop_config.json` file:

**Location:**

- MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`

**Configuration:**

```json
{
  "mcpServers": {
    "omop_mcp": {
      "command": "uv",
      "args": ["--directory", "<path-to-local-repo>", "run", "omop_mcp"]
    }
  }
}
```

## Features

The OMOP MCP server provides the `find_omop_concept` tool for:

- Mapping clinical terminology to OMOP concepts
- Validating terminology mappings
- Searching OMOP vocabulary
- Converting between different clinical coding systems

## Usage Example

- It is recommended to specify the OMOP field and table name in the prompt for improved accuracy.
  Refer to [omop_concept_id_fields.json](src/omop_mcp/data/omop_concept_id_fields.json) for the list of OMOP fields and tables that store concept IDs.

- You can specify preferred vocabularies for the mapping in order of priority (e.g., "SNOMED preferred" or "LOINC > SNOMED > RxNorm").

**Prompt:**
"Map `Temperature Temporal Scanner - RR` for `measurement_concept_id` in the `measurement` table."

**Response:**

```json
{
  "id": 46235152,
  "code": "75539-7",
  "name": "Body temperature - Temporal artery",
  "class": "Clinical Observation",
  "concept": "Standard",
  "validity": "Valid",
  "domain": "Measurement",
  "vocab": "LOINC",
  "url": "https://athena.ohdsi.org/search-terms/terms/46235152",
  "processing_time_sec": "0.601"
}
```
