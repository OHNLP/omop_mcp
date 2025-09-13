[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/ohnlp-omop-mcp-badge.png)](https://mseep.ai/app/ohnlp-omop-mcp)

# OMOP MCP Server

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2509.03828-b31b1b.svg)](https://arxiv.org/abs/2509.03828)

Model Context Protocol (MCP) server for mapping clinical terminology to Observational Medical Outcomes Partnership (OMOP) concepts using Large Language Models (LLMs).

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

```
CONCEPT_ID: 46235152
CODE: 75539-7
NAME: Body temperature - Temporal artery
CLASS: Clinical Observation
CONCEPT: Standard
VALIDITY: Valid
DOMAIN: Measurement
VOCAB: LOINC
URL: https://athena.ohdsi.org/search-terms/terms/46235152
REASON: This LOINC concept specifically represents body temperature measured at the temporal artery, which is what a temporal scanner measures. The "RR" in your source term likely refers to "Recovery Room" or another location/department indicator, but in OMOP, the location would typically be captured in a separate field rather than as part of the measurement concept itself.
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTION.md) for guidelines to contribute to the project.

## Citation Policy

If you use this software, please cite the pre-print at arXiv (cs.AI) below:

[An Agentic Model Context Protocol Framework for Medical Concept Standardization](https://arxiv.org/abs/2509.03828)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

**Contact:** jaerongahn@gmail.com
