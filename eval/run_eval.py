"""Run the agent on eval/tasks.jsonl and merge the numbers into results.json.

Variants
  no_tools    the same LLM answers from its weights (baseline)
  agent       agent loop with tools imported directly
  agent_mcp   the same loop, tools served by the MCP server over stdio (should match `agent`)

Metrics: success rate (rule check vs gold), tool calls per task, share of tool calls that returned an error,
malformed tool calls, tasks that hit max_steps, latency.

Run:  CTGOV_CACHE=record uv run python eval/run_eval.py [--variants no_tools agent agent_mcp] [--limit 10]
"""

import argparse
import json
import os
import platform
import re
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.llm import get_chat  # noqa: E402
from agent.loop import DirectToolBox, run_agent, run_no_tools  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, RUNS = ROOT / "results.json", ROOT / "eval" / "runs"
_STOP = {"the", "of", "and", "for", "inc", "ltd", "llc", "co", "company", "corporation", "a", "an", "srl", "gmbh"}


def numbers(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", text.replace(",", ""))}


def name_match(gold: str, answer: str, min_recall: float = 0.6) -> bool:
    g = {t for t in re.findall(r"[a-z0-9]+", gold.lower()) if t not in _STOP}
    a = set(re.findall(r"[a-z0-9]+", answer.lower()))
    return bool(g) and len(g & a) / len(g) >= min_recall


def is_success(task: dict, answer: str) -> bool:
    gold = task["gold"]
    match task["type"]:
        case "count" | "lookup" | "aggregate":
            return gold in numbers(answer)
        case "earliest":
            return gold in answer.upper()
        case "sponsor":
            return name_match(gold, answer)
        case "compare":
            return gold["winner"] in answer.upper() and gold["diff"] in numbers(answer)
    raise ValueError(task["type"])


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))])


def merge(section: str, data: dict) -> None:
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res[section] = data
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["no_tools", "agent", "agent_mcp"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="", help="suffix for the result key, e.g. v2 after changing tool descriptions")
    args = ap.parse_args()
    # record: answer from the committed cache, fetch + store anything new. A pure replay cache made every query the
    # task builder hadn't anticipated (e.g. max_results=10 instead of 25) fail, which punished valid agent behaviour.
    os.environ.setdefault("CTGOV_CACHE", "record")
    tasks = [json.loads(line) for line in (ROOT / "eval" / "tasks.jsonl").open()]
    if args.limit:
        tasks = tasks[:: max(1, len(tasks) // args.limit)]
    llm = get_chat()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = json.loads(RESULTS.read_text()).get("agent", {}) if RESULTS.exists() else {}
    out["meta"] = {"date": datetime.now(UTC).isoformat(timespec="seconds"), "llm": llm.name,
                   "hardware": f"{platform.system()} {platform.machine()}", "n_tasks": len(tasks),
                   "cache": os.environ["CTGOV_CACHE"]}  # fmt: skip
    for v in args.variants:
        key = f"{v}_{args.tag}" if args.tag else v
        tools = None
        if v == "agent":
            tools = DirectToolBox()
        elif v == "agent_mcp":
            from agent.mcp_tools import MCPToolBox

            tools = MCPToolBox()
        recs = []
        for t in tasks:
            tr = run_no_tools(t["task"], llm) if tools is None else run_agent(t["task"], llm, tools)
            recs.append({"id": t["id"], "type": t["type"], "success": is_success(t, tr.answer), **tr.to_dict()})
            print(key, t["id"], t["type"], recs[-1]["success"], tr.n_tool_calls, repr(tr.answer[:100]), flush=True)
        if v == "agent_mcp":
            tools.close()
        (RUNS / f"{key}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        n_calls = sum(r["n_tool_calls"] for r in recs)
        out[key] = {
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cache_mode": os.environ["CTGOV_CACHE"],
            "success": round(statistics.mean(r["success"] for r in recs), 3),
            "tool_calls_per_task": round(n_calls / len(recs), 2),
            "tool_error_rate": round(sum(r["n_tool_errors"] for r in recs) / n_calls, 3) if n_calls else None,
            "malformed_calls": sum(len(s["parse_errors"]) for r in recs for s in r["steps"]),
            "hit_max_steps": sum(r["stopped"] == "max_steps" for r in recs),
            "latency_s_p50": pct([r["total_ms"] / 1000 for r in recs], 0.5),
            "latency_s_p95": pct([r["total_ms"] / 1000 for r in recs], 0.95),
            "by_type": {
                k: round(statistics.mean(r["success"] for r in recs if r["type"] == k), 3)
                for k in sorted({r["type"] for r in recs})
            },  # fmt: skip
        }
        print(key, {k: x for k, x in out[key].items() if k != "by_type"}, flush=True)
        merge("agent", out)


if __name__ == "__main__":
    main()
