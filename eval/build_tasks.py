"""Build agent tasks whose gold answers are computed by code with the same tools (so they are exact).

Six task types, from one tool call to four:
  lookup     get_trial                                  "What was the enrollment of NCT…?"
  count      count_trials                               "How many completed Phase 3 trials of X for Y?"
  earliest   search_trials                              "Which … started first? Give its NCT id."
  sponsor    search_trials (+ get_trial)                "Who sponsored the earliest …?"
  compare    get_trial ×2 + calculator                  "Which of A and B enrolled more, and by how many?"
  aggregate  search_trials + calculator                 "Total enrollment of all completed Phase 3 trials of X for Y?"

Run with CTGOV_CACHE=record once (fills eval/cache/), afterwards everything replays offline.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ctgov.tools import count_trials, get_trial, search_trials  # noqa: E402

SEED = 0
PAIRS = [
    ("bimekizumab", "psoriasis"), ("secukinumab", "psoriasis"), ("ixekizumab", "psoriasis"),
    ("semaglutide", "type 2 diabetes"), ("empagliflozin", "type 2 diabetes"), ("tirzepatide", "type 2 diabetes"),
    ("dupilumab", "asthma"), ("mepolizumab", "asthma"), ("benralizumab", "asthma"),
    ("lecanemab", "alzheimer disease"), ("donanemab", "alzheimer disease"),
    ("tofacitinib", "rheumatoid arthritis"), ("baricitinib", "rheumatoid arthritis"),
    ("upadacitinib", "rheumatoid arthritis"),
    ("esketamine", "depression"), ("vortioxetine", "depression"),
]  # fmt: skip
PER_TYPE = 6
FILTER = {"phase": "PHASE3", "status": "COMPLETED"}


def main() -> None:
    rng = random.Random(SEED)
    info = {}
    for drug, cond in PAIRS:
        res = search_trials(condition=cond, intervention=drug, max_results=25, **FILTER)
        trials = [t for t in res["trials"] if t["enrollment"] is not None and t["start_date"]]
        info[(drug, cond)] = (res["total"], trials)
        print(f"{drug:14s} {cond:22s} total={res['total']:3d} listed={len(trials)}")

    phr = "completed Phase 3 trials of {d} for {c}"
    tasks: list[dict] = []
    usable = [p for p in PAIRS if info[p][1]]
    for drug, cond in rng.sample(usable, PER_TYPE):
        total = count_trials(condition=cond, intervention=drug, **FILTER)["total"]
        q = f"How many {phr.format(d=drug, c=cond)} are registered on ClinicalTrials.gov?"
        tasks.append({"type": "count", "task": q, "gold": total})
    for drug, cond in rng.sample(usable, PER_TYPE):
        t = rng.choice(info[(drug, cond)][1])
        tasks.append({"type": "lookup", "task": f"How many participants were enrolled in trial {t['nct_id']}?",
                      "gold": get_trial(t["nct_id"])["enrollment"]})  # fmt: skip
    small = [p for p in usable if info[p][0] <= 25]  # the tool lists at most 25, so "first" must be complete
    for drug, cond in rng.sample(small, PER_TYPE):
        first = info[(drug, cond)][1][0]
        q = f"Which of the {phr.format(d=drug, c=cond)} started first? Give its NCT id."
        tasks.append({"type": "earliest", "task": q, "gold": first["nct_id"]})
    for drug, cond in rng.sample(small, PER_TYPE):
        first = info[(drug, cond)][1][0]
        q = f"Who is the lead sponsor of the earliest-starting of the {phr.format(d=drug, c=cond)}?"
        tasks.append({"type": "sponsor", "task": q, "gold": first["lead_sponsor"]})
    two = [p for p in usable if len(info[p][1]) >= 2]
    for drug, cond in rng.sample(two, PER_TYPE):
        a, b = rng.sample(info[(drug, cond)][1], 2)
        ea, eb = get_trial(a["nct_id"])["enrollment"], get_trial(b["nct_id"])["enrollment"]
        if ea == eb:
            continue
        q = f"Which of the trials {a['nct_id']} and {b['nct_id']} enrolled more participants, and by how many?"
        tasks.append({"type": "compare", "task": q,
                      "gold": {"winner": a["nct_id"] if ea > eb else b["nct_id"], "diff": abs(ea - eb)}})  # fmt: skip
    agg = [p for p in usable if 2 <= info[p][0] <= 8 and len(info[p][1]) == info[p][0]]
    for drug, cond in rng.sample(agg, min(PER_TYPE, len(agg))):
        q = f"What is the total enrollment of all {phr.format(d=drug, c=cond)}?"
        tasks.append({"type": "aggregate", "task": q, "gold": sum(t["enrollment"] for t in info[(drug, cond)][1])})

    for i, t in enumerate(tasks):
        t["id"] = f"t{i:02d}"
    out = Path(__file__).parent / "tasks.jsonl"
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
    print(f"{len(tasks)} tasks → {out}")


if __name__ == "__main__":
    main()
