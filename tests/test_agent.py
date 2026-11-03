"""Fast, offline tests (API answers come from the committed replay cache, no LLM needed)."""

import os

os.environ["CTGOV_CACHE"] = "replay"

import pytest  # noqa: E402

from agent.llm import ToolCall, Turn, parse_tool_calls  # noqa: E402
from agent.loop import DirectToolBox, run_agent  # noqa: E402
from ctgov.tools import call_tool, tool_schemas  # noqa: E402
from eval.run_eval import is_success  # noqa: E402

NCT = "NCT06750289"  # 130 participants, in the replay cache (task t06)


def test_schemas_come_from_type_hints():
    s = {x["function"]["name"]: x["function"] for x in tool_schemas()}
    assert set(s) == {"search_trials", "count_trials", "get_trial", "calculator"}
    assert s["get_trial"]["parameters"]["required"] == ["nct_id"]
    phase = s["search_trials"]["parameters"]["properties"]["phase"]
    assert "PHASE3" in str(phase)


def test_tools_return_errors_as_data():
    assert call_tool("get_trial", {"nct_id": NCT})["enrollment"] == 130
    assert "error" in call_tool("get_trial", {"nct_id": "123"})
    assert "error" in call_tool("search_trials", {"phase": "PHASE9", "condition": "x"})  # not in the enum
    assert "error" in call_tool("nope", {})


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "a + 1", "[1, 2]"])
def test_calculator_refuses_anything_but_arithmetic(expr):
    assert "error" in call_tool("calculator", {"expression": expr})


def test_calculator_math():
    assert call_tool("calculator", {"expression": "(553 + 133) / 2"}) == {"value": 343.0}


def test_parse_qwen_tool_calls():
    t = parse_tool_calls(
        'Let me check.\n<tool_call>\n{"name": "get_trial", "arguments": {"nct_id": "NCT1"}}\n</tool_call>'
    )
    assert t.content == "Let me check." and t.tool_calls[0].name == "get_trial"
    bad = parse_tool_calls("<tool_call>{not json}</tool_call>")
    assert not bad.tool_calls and bad.parse_errors


class ScriptedLLM:
    """Plays back a fixed sequence of turns, so the loop can be tested without a model."""

    name = "scripted"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def chat(self, messages, tools):
        self.seen.append(messages[-1])
        return self.turns.pop(0)


def test_agent_loop_runs_tools_and_feeds_back_errors():
    llm = ScriptedLLM([
        Turn("", [ToolCall("get_trial", {"nct_id": "bad"})]),
        Turn("", [ToolCall("get_trial", {"nct_id": NCT})]),
        Turn(f"{NCT} enrolled 130 participants."),
    ])  # fmt: skip
    tr = run_agent("How many in NCT06750289?", llm, DirectToolBox())
    assert tr.answer.endswith("130 participants.") and tr.n_tool_calls == 2 and tr.n_tool_errors == 1
    assert llm.seen[1]["role"] == "tool" and "error" in llm.seen[1]["content"]  # the model saw its mistake


def test_agent_stops_at_max_steps():
    llm = ScriptedLLM([Turn("", [ToolCall("calculator", {"expression": "1+1"})])] * 3)
    tr = run_agent("loop forever", llm, DirectToolBox(), max_steps=3)
    assert tr.stopped == "max_steps" and tr.answer == ""


def test_mcp_server_exposes_the_same_tools():
    from agent.mcp_tools import MCPToolBox
    from mcp_server.server import mcp

    tb = MCPToolBox(mcp)  # in-memory transport: same protocol, no subprocess
    try:
        assert {s["function"]["name"] for s in tb.schemas()} == {s["function"]["name"] for s in tool_schemas()}
        assert tb.call("get_trial", {"nct_id": NCT})["enrollment"] == 130
        err = tb.call("get_trial", {"nct_id": "bad"})
        assert "NCT01234567" in err["error"]  # the helpful message survives the protocol
    finally:
        tb.close()


@pytest.mark.parametrize(
    ("task", "answer", "ok"),
    [
        ({"type": "count", "gold": 12}, "There are 12 completed trials.", True),
        ({"type": "compare", "gold": {"winner": "NCT1", "diff": 420}}, "NCT1 enrolled 420 more.", True),
        ({"type": "compare", "gold": {"winner": "NCT1", "diff": 420}}, "NCT2 enrolled 420 more.", False),
        ({"type": "sponsor", "gold": "UCB Biopharma SRL"}, "UCB Biopharma", True),
        ({"type": "earliest", "gold": "NCT03410992"}, "nct03410992 started first", True),
    ],
)
def test_success_rules(task, answer, ok):
    assert is_success(task, answer) is ok
