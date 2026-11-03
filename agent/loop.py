"""The agent loop, from scratch: ask the model → run the tools it asked for → feed results back → repeat.

That's all an "agent" is. Everything else is guard rails:
- max_steps, so a confused model can't loop forever (and burn money with a paid API),
- tool errors returned as data, so the model can fix its own call,
- tool output truncated, so one huge result can't push the question out of the context window,
- a full trace of every step, because you can't debug (or audit) an agent you can't see.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

from agent.llm import ChatLLM

MAX_TOOL_CHARS = 4000

SYSTEM = """You are a research assistant for clinical trials with tools for ClinicalTrials.gov.
- Look facts up with the tools; never guess trial facts or numbers.
- Use the calculator for every sum, difference or average.
- When you have the answer, reply with a short final answer that cites the NCT ids you used.
- If the tools do not give you the answer, say "I don't know"."""

SYSTEM_NO_TOOLS = """You are a research assistant for clinical trials.
Answer briefly. If you do not know, say "I don't know"."""


class ToolBox(Protocol):
    def schemas(self) -> list[dict]: ...

    def call(self, name: str, arguments: dict) -> dict: ...


class DirectToolBox:
    """Tools imported as Python functions (the agent and the tools live in one process)."""

    def schemas(self) -> list[dict]:
        from ctgov.tools import tool_schemas

        return tool_schemas()

    def call(self, name: str, arguments: dict) -> dict:
        from ctgov.tools import call_tool

        return call_tool(name, arguments)


@dataclass
class Step:
    content: str
    tool_calls: list[dict]
    results: list[dict]
    parse_errors: list[str]
    llm_ms: float
    tool_ms: float


@dataclass
class Trace:
    task: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    stopped: str = "answer"  # answer | max_steps
    total_ms: float = 0.0

    @property
    def n_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def n_tool_errors(self) -> int:
        return sum(1 for s in self.steps for r in s.results if "error" in r)

    def to_dict(self) -> dict:
        return asdict(self) | {"n_tool_calls": self.n_tool_calls, "n_tool_errors": self.n_tool_errors}


def run_agent(task: str, llm: ChatLLM, tools: ToolBox, max_steps: int = 8) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(task)
    schemas = tools.schemas()
    messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(max_steps):
        t = time.perf_counter()
        turn = llm.chat(messages, schemas)
        llm_ms = (time.perf_counter() - t) * 1000
        if not turn.tool_calls:
            trace.steps.append(Step(turn.content, [], [], turn.parse_errors, llm_ms, 0.0))
            if turn.parse_errors:  # it tried to call a tool but broke the format: tell it and let it retry
                messages.append({"role": "assistant", "content": turn.content})
                messages.append({"role": "user", "content": "Your tool call was not valid JSON. Try again."})
                continue
            trace.answer = turn.content
            break
        calls = [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls]
        messages.append(
            {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            }
        )
        t = time.perf_counter()
        results = []
        for c in turn.tool_calls:
            res = tools.call(c.name, c.arguments)
            results.append(res)
            text = json.dumps(res, ensure_ascii=False)
            if len(text) > MAX_TOOL_CHARS:
                text = text[:MAX_TOOL_CHARS] + ' ... [truncated: ask for fewer results]"'
            messages.append({"role": "tool", "content": text, "tool_name": c.name})
        trace.steps.append(
            Step(turn.content, calls, results, turn.parse_errors, llm_ms, (time.perf_counter() - t) * 1000)
        )
    else:
        trace.stopped = "max_steps"
    trace.total_ms = (time.perf_counter() - t0) * 1000
    return trace


def run_no_tools(task: str, llm: ChatLLM) -> Trace:
    """Baseline: the same model answers from its weights alone."""
    t0 = time.perf_counter()
    turn = llm.chat([{"role": "system", "content": SYSTEM_NO_TOOLS}, {"role": "user", "content": task}], None)
    ms = (time.perf_counter() - t0) * 1000
    return Trace(task, turn.content, [Step(turn.content, [], [], [], ms, 0.0)], "answer", ms)
