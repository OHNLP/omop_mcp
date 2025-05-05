import httpx
import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl

# Store notes as a simple key-value dict to demonstrate state management
notes: dict[str, str] = {}

server = Server("omop_mcp")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available note resources.
    Each note is exposed as a resource with a custom note:// URI scheme.
    """
    return [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific note's content by its URI.
    The note name is extracted from the URI host component.
    """
    if uri.scheme != "note":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    name = uri.path
    if name is not None:
        name = name.lstrip("/")
        return notes[name]
    raise ValueError(f"Note not found: {name}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    if name != "summarize-notes":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "brief")
    detail_prompt = " Give extensive details." if style == "detailed" else ""

    return types.GetPromptResult(
        description="Summarize the current notes",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                    + "\n".join(
                        f"- {name}: {content}" for name, content in notes.items()
                    ),
                ),
            )
        ],
    )


async def query_athena(keyword: str):
    """Query Athena OHDSI API for a concept matching the keyword."""
    url = "https://athena.ohdsi.org/api/v1/concepts"
    params = {"query": keyword}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]  # Return the best match
        return None


# Simple mapping for OMOP table/field to domain/vocab
OMOP_DOMAIN_VOCAB = {
    ("measurement", None): ("Measurement", "LOINC"),
    ("condition_occurrence", None): ("Condition", "SNOMED"),
    ("drug_exposure", None): ("Drug", "RxNorm"),
    ("procedure_occurrence", None): ("Procedure", "SNOMED"),
    ("observation", None): ("Observation", "SNOMED"),
    # Add more as needed
}


def get_domain_vocab(table: str, field: str | None = None):
    return OMOP_DOMAIN_VOCAB.get((table.lower(), field), ("Unknown", "Unknown"))


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    tools = [
        types.Tool(
            name="map_medical_keyword",
            description="Map a medical keyword to OMOP concept using Athena and OMOP CDM.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "table": {"type": "string"},
                    "field": {"type": "string"},
                },
                "required": ["keyword", "table", "field"],
            },
        ),
        types.Tool(
            name="add-note",
            description="Add a new note",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        ),
    ]
    return tools


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name == "map_medical_keyword":
        if not arguments:
            raise ValueError("Missing arguments")
        keyword = arguments.get("keyword")
        table = arguments.get("table")
        field = arguments.get("field")
        if not keyword or not table or not field:
            raise ValueError("Missing keyword, table, or field")
        concept = await query_athena(keyword)
        domain, vocab = get_domain_vocab(table, field)
        if not concept:
            return [
                types.TextContent(
                    type="text", text=f"No OMOP concept found for '{keyword}'."
                )
            ]
        # Compose the output as a markdown table
        output = (
            "| ID | CODE | NAME | CLASS | CONCEPT | VALIDITY | DOMAIN | VOCAB |\n"
            "|----|------|------|-------|---------|----------|--------|-------|\n"
            f"| {concept.get('id', '')} | {concept.get('code', '')} | "
            f"{concept.get('name', '')} | {concept.get('classId', '')} | "
            f"{concept.get('standardConcept', '')} | "
            f"{concept.get('validity', '')} | "
            f"{domain} | {vocab} |"
        )
        return [types.TextContent(type="text", text=output)]
    elif name == "add-note":
        if not arguments:
            raise ValueError("Missing arguments")
        note_name = arguments.get("name")
        content = arguments.get("content")
        if not note_name or not content:
            raise ValueError("Missing name or content")
        notes[note_name] = content
        await server.request_context.session.send_resource_list_changed()
        return [
            types.TextContent(
                type="text",
                text=f"Added note '{note_name}' with content: {content}",
            )
        ]
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="omop_mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
