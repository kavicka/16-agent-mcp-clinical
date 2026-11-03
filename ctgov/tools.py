"""The four tools the agent (and the MCP server) can use.

Design rules that matter for a small LLM:
- few tools with clear names and docstrings: the docstring IS the prompt the model reads to pick a tool,
- flat, typed arguments (validated by pydantic, so a bad call becomes a readable error, not a crash),
- short JSON results with only the fields a question needs: every token of output goes back into the context.
"""

import ast
import inspect
import operator
from collections.abc import Callable
from typing import Any, Literal

from pydantic import ValidationError, create_model

from ctgov import client

Phase = Literal["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
Status = Literal["COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING", "TERMINATED", "WITHDRAWN", "NOT_YET_RECRUITING"]
LIST_FIELDS = "NCTId,BriefTitle,Phase,OverallStatus,EnrollmentCount,LeadSponsorName,StartDate"


def _query(condition: str | None, intervention: str | None, phase: str | None, status: str | None, sponsor: str | None):
    q: dict[str, Any] = {}
    if condition:
        q["query.cond"] = condition
    if intervention:
        q["query.intr"] = intervention
    if sponsor:
        q["query.spons"] = sponsor
    if status:
        q["filter.overallStatus"] = status
    if phase:
        q["filter.advanced"] = f"AREA[Phase]{phase}"
    return q


def _brief(study: dict) -> dict:
    p = study["protocolSection"]
    return {
        "nct_id": p["identificationModule"]["nctId"],
        "title": p["identificationModule"].get("briefTitle"),
        "phases": p.get("designModule", {}).get("phases", []),
        "status": p.get("statusModule", {}).get("overallStatus"),
        "enrollment": p.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
        "lead_sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        "start_date": p.get("statusModule", {}).get("startDateStruct", {}).get("date"),
    }


def search_trials(
    condition: str | None = None,
    intervention: str | None = None,
    phase: Phase | None = None,
    status: Status | None = None,
    sponsor: str | None = None,
    max_results: int = 10,
) -> dict:
    """Search ClinicalTrials.gov. Returns the total number of matches and up to max_results trials
    (nct_id, title, phases, status, enrollment, lead_sponsor, start_date), oldest start date first.
    Give at least one of condition, intervention or sponsor."""
    q = _query(condition, intervention, phase, status, sponsor)
    if not any(k.startswith("query.") for k in q):
        raise ValueError("give at least one of condition, intervention or sponsor")
    params = {**q, "fields": LIST_FIELDS, "pageSize": max(1, min(max_results, 25)), "countTotal": "true",
              "sort": "StartDate:asc"}  # fmt: skip
    data = client.get(params)
    return {"total": data.get("totalCount", 0), "trials": [_brief(s) for s in data.get("studies", [])]}


def count_trials(
    condition: str | None = None,
    intervention: str | None = None,
    phase: Phase | None = None,
    status: Status | None = None,
    sponsor: str | None = None,
) -> dict:
    """Count trials on ClinicalTrials.gov matching the filters, without listing them."""
    q = _query(condition, intervention, phase, status, sponsor)
    if not any(k.startswith("query.") for k in q):
        raise ValueError("give at least one of condition, intervention or sponsor")
    data = client.get({**q, "fields": "NCTId", "pageSize": 1, "countTotal": "true"})
    return {"total": data.get("totalCount", 0)}


def get_trial(nct_id: str) -> dict:
    """Details of one trial by its NCT id (e.g. NCT01234567): phases, status, enrollment, sponsor, dates,
    conditions, interventions, primary outcomes and age limits."""
    nct_id = nct_id.strip().upper()
    if not (nct_id.startswith("NCT") and len(nct_id) == 11 and nct_id[3:].isdigit()):
        raise ValueError("nct_id must look like NCT01234567")
    data = client.get({"filter.ids": nct_id, "pageSize": 1, "fields": "protocolSection"})
    if not data.get("studies"):
        raise LookupError(f"{nct_id} not found")
    s = data["studies"][0]
    p = s["protocolSection"]
    return {
        **_brief(s),
        "completion_date": p.get("statusModule", {}).get("completionDateStruct", {}).get("date"),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "interventions": [i.get("name") for i in p.get("armsInterventionsModule", {}).get("interventions", [])],
        "primary_outcomes": [o.get("measure") for o in p.get("outcomesModule", {}).get("primaryOutcomes", [])][:3],
        "minimum_age": p.get("eligibilityModule", {}).get("minimumAge"),
        "maximum_age": p.get("eligibilityModule", {}).get("maximumAge"),
    }


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.Pow: operator.pow}  # fmt: skip


def calculator(expression: str) -> dict:
    """Evaluate an arithmetic expression with + - * / ** and parentheses, e.g. "(553 + 133) / 2".
    Use it for every sum, difference or average instead of computing in your head."""

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, int | float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / ** ( ) are allowed")  # no names, calls or attributes: no eval()

    value = ev(ast.parse(expression, mode="eval").body)
    return {"value": round(value, 6) if isinstance(value, float) else value}


TOOLS: dict[str, Callable[..., dict]] = {f.__name__: f for f in (search_trials, count_trials, get_trial, calculator)}


def _args_model(fn: Callable):
    fields = {
        name: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
        for name, p in inspect.signature(fn).parameters.items()
    }
    return create_model(f"{fn.__name__}_args", **fields)


ARGS = {name: _args_model(fn) for name, fn in TOOLS.items()}


def tool_schemas() -> list[dict]:
    """OpenAI/Ollama-style function schemas, generated from the type hints so code and schema can't drift."""
    out = []
    for name, fn in TOOLS.items():
        schema = ARGS[name].model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        out.append(
            {"type": "function", "function": {"name": name, "description": inspect.getdoc(fn), "parameters": schema}}
        )
    return out


def call_tool(name: str, arguments: dict) -> dict:
    """Validate and run a tool. Errors come back as data, so the model can read them and try again."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}; available: {', '.join(TOOLS)}"}
    try:
        args = ARGS[name].model_validate(arguments or {})
        return TOOLS[name](**args.model_dump())
    except ValidationError as e:
        return {
            "error": "invalid arguments: "
            + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        }
    except (ValueError, LookupError, SyntaxError, ZeroDivisionError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
