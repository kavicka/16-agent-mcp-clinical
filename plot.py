"""results.json + eval/runs/*.jsonl → assets/*.png.

success.png  success rate per task type, LLM alone vs agent (vs agent over MCP)
trace.png    the step's visual example: one agent run drawn as a timeline (LLM thinking vs tool calls)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = json.loads(Path("results.json").read_text())["agent"]
Path("assets").mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"
COLORS = {"no_tools": GRAY, "agent": "#93c5fd", "agent_mcp": "#86efac", "agent_v2": BLUE, "agent_mcp_v2": GREEN}
LABELS = {
    "no_tools": "LLM alone",
    "agent": "agent, tools v1",
    "agent_mcp": "agent over MCP, tools v1",
    "agent_v2": "agent, tools v2",
    "agent_mcp_v2": "agent over MCP, tools v2",
}
TYPES = ["lookup", "count", "earliest", "sponsor", "compare", "aggregate"]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def success() -> None:
    variants = [v for v in COLORS if v in R]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(TYPES) + 1)
    w = 0.8 / len(variants)
    for j, v in enumerate(variants):
        vals = [R[v]["by_type"].get(t, 0) for t in TYPES] + [R[v]["success"]]
        bars = ax.bar(x + (j - (len(variants) - 1) / 2) * w, vals, w, color=COLORS[v], label=LABELS[v])
        ax.bar_label(bars, fmt="%.2f", fontsize=5, rotation=90, padding=2)
    ax.set(xticks=x, xticklabels=[*TYPES, "ALL"], ylim=(0, 1.3), ylabel="success rate")
    ax.set_title(f"{R['meta']['n_tasks']} tasks · {R['meta']['llm']} · {R['meta']['hardware']}", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", ncol=5)
    ax.grid(alpha=0.3, axis="y")
    save(fig, "success.png")


def trace() -> None:
    """One agent run as a sequence of messages: the step's visual example of what "an agent" actually does."""
    runs = Path("eval/runs/agent_v2.jsonl")
    if not runs.exists():
        return
    recs = [json.loads(line) for line in runs.open()]
    # the most instructive run: a successful one that used the most tool calls
    rec = max((r for r in recs if r["success"]), key=lambda r: (r["n_tool_calls"], -r["total_ms"]), default=None)
    if rec is None:
        return
    events = [("user", GRAY, rec["task"])]
    for s in rec["steps"]:
        for j, (c, r) in enumerate(zip(s["tool_calls"], s["results"], strict=True)):
            args = ", ".join(f"{k}={v!r}" for k, v in c["arguments"].items())
            when = f"({s['llm_ms'] / 1000:.1f} s)" if j == 0 else "(same turn)"  # parallel calls come in one turn
            events.append((f"LLM → tool  {when}", BLUE, f"{c['name']}({args})"))
            brief = {k: r[k] for k in ("nct_id", "enrollment", "lead_sponsor", "total", "value") if k in r}
            events.append(("tool → LLM", ORANGE, json.dumps(brief or r)[:110]))
        if not s["tool_calls"]:
            events.append((f"LLM answer  ({s['llm_ms'] / 1000:.1f} s)", GREEN, s["content"]))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(events) + 0.6))
    for i, (who, color, text) in enumerate(events):
        y = len(events) - i
        ax.text(0.01, y, who, fontsize=8, color="white", va="center",
                bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": color})  # fmt: skip
        ax.text(0.22, y, text if len(text) < 120 else text[:117] + "…", fontsize=8, va="center", family="monospace")
    ax.set(xlim=(0, 1), ylim=(0.4, len(events) + 0.6))
    ax.axis("off")
    ax.set_title(f"One agent run, {rec['total_ms'] / 1000:.0f} s on CPU", fontsize=10, loc="left")
    save(fig, "trace.png")


if __name__ == "__main__":
    success()
    trace()
