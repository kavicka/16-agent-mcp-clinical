"""The same four tools, exposed as an MCP server (Model Context Protocol).

Why MCP: written once, the tools work in any MCP client (Claude Code, Claude Desktop, IDEs, our own agent)
without glue code per client. The server speaks JSON-RPC over stdio: the client starts it as a subprocess.

Run / register:
    uv run python -m mcp_server.server                                         # stdio server
    claude mcp add clinical-trials -- uv run --directory "$PWD" python -m mcp_server.server
"""

import functools
from collections.abc import Callable

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ctgov.tools import TOOLS

mcp = MCPServer("clinical-trials", instructions="Search and read ClinicalTrials.gov records.", log_level="WARNING")


def _expected_errors_as_tool_errors(fn: Callable) -> Callable:
    """MCP hides unexpected exceptions behind "Error executing tool …". A bad NCT id is not a crash, it's
    feedback the model needs to fix its call, so expected errors are re-raised as ToolError with their message."""

    @functools.wraps(fn)  # keeps name, docstring and signature, so the generated schema stays the same
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, LookupError, SyntaxError, ZeroDivisionError, httpx.HTTPError) as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


for fn in TOOLS.values():
    # MCPServer (the MCP SDK 2.x name of FastMCP) builds the JSON schema from the type hints and uses the docstring
    # as the description, exactly like tool_schemas() does for the direct agent: both see the same definitions.
    mcp.add_tool(_expected_errors_as_tool_errors(fn))


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
