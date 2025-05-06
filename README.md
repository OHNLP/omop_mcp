# OMOP MCP Server

Model Context Protocol (MCP) server for mapping clinical terminology to OMOP concepts using Large Language Models (LLMs).

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

**Prompt:**
"Map 'Rehab' for the 'discharge_to_concept_id' field in the 'visit_occurrence' table"

**Request:**

```json
{
  "keyword": "discharge to rehabilitation",
  "omop_field": "discharge_to_concept_id",
  "omop_table": "visit_occurrence"
}
```

**Response:**

```json
{
  "id": 762906,
  "code": "433591000124103",
  "name": "Discharge to rehabilitation facility",
  "class": "Procedure",
  "concept": "Standard",
  "validity": "Valid",
  "domain": "Observation",
  "vocab": "SNOMED"
}
```
