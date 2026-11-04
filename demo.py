"""Run one task and print the agent's trace step by step: what it asked for, what it got back, what it answered.

Run:  uv run python demo.py "your question"            # tools imported directly
      uv run python demo.py --mcp "your question"      # tools through the MCP server
"""

import argparse
import json

from agent.llm import get_chat
from agent.loop import DirectToolBox, run_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--mcp", action="store_true")
    args = ap.parse_args()
    if args.mcp:
        from agent.mcp_tools import MCPToolBox

        tools = MCPToolBox()
    else:
        tools = DirectToolBox()
    tr = run_agent(args.task, get_chat(), tools)
    for i, s in enumerate(tr.steps, 1):
        print(f"\n── step {i} · LLM {s.llm_ms / 1000:.1f} s")
        if s.content:
            print("model:", s.content)
        for c, r in zip(s.tool_calls, s.results, strict=True):
            print(f"→ {c['name']}({json.dumps(c['arguments'])})")
            print("←", json.dumps(r, ensure_ascii=False)[:300])
    print(f"\nanswer ({tr.stopped}, {tr.total_ms / 1000:.1f} s): {tr.answer}")
    if args.mcp:
        tools.close()


if __name__ == "__main__":
    main()
