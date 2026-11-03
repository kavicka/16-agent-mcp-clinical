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
COLORS = {"no_tools": GRAY, "agent": BLUE, "agent_mcp": GREEN}
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
        bars = ax.bar(x + (j - (len(variants) - 1) / 2) * w, vals, w, color=COLORS[v], label=v)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set(xticks=x, xticklabels=[*TYPES, "ALL"], ylim=(0, 1.12), ylabel="success rate")
    ax.set_title(f"{R['meta']['n_tasks']} tasks · {R['meta']['llm']} · {R['meta']['hardware']}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "success.png")


def trace() -> None:
    runs = Path("eval/runs/agent.jsonl")
    if not runs.exists():
        return
    recs = [json.loads(line) for line in runs.open()]
    # the most instructive run: a successful one that used the most tool calls
    rec = max((r for r in recs if r["success"]), key=lambda r: (r["n_tool_calls"], -r["total_ms"]), default=None)
    if rec is None:
        return
    fig, ax = plt.subplots(figsize=(10, 2.6 + 0.25 * len(rec["steps"])))
    t, y = 0.0, 0
    for i, s in enumerate(rec["steps"], 1):
        ax.barh(y, s["llm_ms"] / 1000, left=t, color=BLUE, height=0.6)
        ax.text(t + s["llm_ms"] / 2000, y, f"LLM {i}", ha="center", va="center", color="white", fontsize=8)
        t += s["llm_ms"] / 1000
        if s["tool_calls"]:
            names = ", ".join(c["name"] for c in s["tool_calls"])
            err = any("error" in r for r in s["results"])
            ax.barh(y, max(s["tool_ms"] / 1000, 0.3), left=t, color=RED if err else ORANGE, height=0.6)
            ax.text(t + 0.4, y - 0.45, names, fontsize=8, va="top")
            t += s["tool_ms"] / 1000
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_title(f"Agent trace: {rec['task'][:90]}\nanswer: {rec['answer'][:110]}", fontsize=9, loc="left")
    ax.set_ylim(-1.2, 0.6)
    save(fig, "trace.png")


if __name__ == "__main__":
    success()
    trace()
