"""ClinicalTrials.gov API v2 client with a record/replay cache.

Why the cache: the registry changes every day, so an eval against the live API would give different gold answers
next week. In `replay` mode every request is answered from eval/cache/ (committed), which makes the agent eval
reproducible and offline. `record` fills the cache from the live API; `live` skips it.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

API = "https://clinicaltrials.gov/api/v2/studies"
CACHE_DIR = Path(__file__).resolve().parent.parent / "eval" / "cache"


def _mode() -> str:
    return os.environ.get("CTGOV_CACHE", "record")  # record | replay | live


def _key(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]


def get(params: dict[str, Any]) -> dict[str, Any]:
    """GET /studies with the given query params, through the cache."""
    path = CACHE_DIR / f"{_key(params)}.json"
    mode = _mode()
    if mode != "live" and path.exists():
        return json.loads(path.read_text())["response"]
    if mode == "replay":
        raise LookupError(f"not in the replay cache: {params}")
    for attempt in range(4):
        try:
            r = httpx.get(API, params=params, timeout=30, headers={"User-Agent": "agent-mcp-clinical (learning)"})
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    data = r.json()
    if mode == "record":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"params": params, "response": data}, ensure_ascii=False, sort_keys=True))
    return data
