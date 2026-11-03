"""MCP client side: the agent gets its tools from an MCP server instead of importing them.

The agent loop doesn't change at all: it still sees `schemas()` and `call()`. That is the point of MCP:
swap where tools come from (our server, a GitHub server, a database server) without touching the agent.
The MCP SDK is async; this wrapper runs it on a background event loop so the agent loop can stay simple.
"""

import asyncio
import json
import os
import sys
import threading

from mcp import Client, StdioServerParameters

# The SDK starts a stdio server with a minimal environment on purpose (a third-party server shouldn't see your
# API keys). Our server needs a few variables, so they are passed explicitly: an allow-list, not os.environ.
SERVER_ENV = ("CTGOV_CACHE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
              "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PATH", "HOME")  # fmt: skip


def server_env() -> dict[str, str]:
    return {k: os.environ[k] for k in SERVER_ENV if k in os.environ}


class MCPToolBox:
    """Starts `python -m mcp_server.server` as a subprocess and talks MCP to it over stdio."""

    def __init__(self, server=None):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        # a StdioServerParameters → real subprocess; an MCPServer object → in-memory (used in tests)
        self._server = server or StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_server.server"], env=server_env()
        )
        self._ready = threading.Event()
        self._session = asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        if not self._ready.wait(timeout=60):
            self._session.result(timeout=1)  # surfaces the start-up error, if any
            raise TimeoutError("MCP server did not start")

    async def _main(self) -> None:
        # One long-lived task owns the connection: the SDK's task groups must be entered and exited
        # in the same task, so opening in one coroutine and closing in another is not allowed.
        self._closing = asyncio.Event()
        async with Client(self._server) as client:  # entering runs the MCP handshake
            self.client = client
            self._tools = (await client.list_tools()).tools
            self._ready.set()
            await self._closing.wait()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=120)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
            }
            for t in self._tools
        ]

    def call(self, name: str, arguments: dict) -> dict:
        res = self._run(self.client.call_tool(name, arguments))
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            return {"error": text}
        sc = res.structured_content
        if isinstance(sc, dict):
            return sc["result"] if set(sc) == {"result"} else sc
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._closing.set)
        self._session.result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
