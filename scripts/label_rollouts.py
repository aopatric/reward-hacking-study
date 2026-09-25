"""Behavioural labels + per-cell table for a planted-flaw run.

    uv run scripts/label_rollouts.py smoke-flaws

Writes runs/<id>/labels.jsonl (one row per rollout, keyed by cell/prompt/sample,
no completion text) and runs/<id>/cell_summary.json. Never rewrites
rollouts.jsonl. Pure CPU; each label runs a handful of sandboxed subprocesses,
so it fans out over a process pool.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeling import EXPLOIT_OUTCOMES, HONEST_OUTCOMES, OUTCOMES, label  # noqa: E402

KEY = ("cell", "prompt_index", "sample_index")


def _label(r: dict) -> dict:
    return {**{k: r[k] for k in KEY}, **label(r)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run = Path("runs") / args.run_id

    rows = [json.loads(line) for line in (run / "rollouts.jsonl").open() if line.strip()]
    with Pool(args.workers) as pool:
        labels = pool.map(_label, rows, chunksize=8)
    with (run / "labels.jsonl").open("w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")

    cells: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for r, lab in zip(rows, labels):
        cells[r["cell"]].append((r, lab))

    summary = {}
    for cell, pairs in cells.items():
        n = len(pairs)
        outcomes = Counter(lab["outcome"] for _, lab in pairs)
        edits = Counter(lab["harness_edit"] for _, lab in pairs)
        per_prompt = defaultdict(lambda: [0, 0])  # [exploits, honest]
        for r, lab in pairs:
            per_prompt[r["prompt_index"]][0] += lab["outcome"] in EXPLOIT_OUTCOMES
            per_prompt[r["prompt_index"]][1] += lab["outcome"] in HONEST_OUTCOMES
        summary[cell] = {
            "n": n,
            "outcomes": {o: outcomes[o] for o in OUTCOMES},
            "f1_triggered": sum(lab["f1_triggered"] for _, lab in pairs),
            "f2_triggered": sum(lab["f2_triggered"] for _, lab in pairs),
            "harness_edit": dict(edits),
            "truncated": sum(r.get("truncated", False) for r, _ in pairs),
            "think_mentions_test": sum(lab["think_mentions_test"] for _, lab in pairs),
            "think_mentions_flaw": sum(lab["think_mentions_flaw"] for _, lab in pairs),
            "prompts_any_exploit": sum(e > 0 for e, _ in per_prompt.values()),
            "prompts_mixed": sum(e > 0 and h > 0 for e, h in per_prompt.values()),
            "num_prompts": len(per_prompt),
        }
    (run / "cell_summary.json").write_text(json.dumps(summary, indent=2))

    hdr = ("cell", "n", "genuine", "hfail", "EXPL", "f1", "f2", "attempt", "authored",
           "malf", "edit", "trunc", "cot:test", "cot:flaw", "mixed/P")
    print(("{:<16}" + "{:>9}" * (len(hdr) - 1)).format(*hdr))
    for cell, s in summary.items():
        o = s["outcomes"]
        print(("{:<16}" + "{:>9}" * (len(hdr) - 1)).format(
            cell, s["n"], o["genuine_solve"], o["honest_fail"], o["exploit_planted"],
            s["f1_triggered"], s["f2_triggered"], o["attempted_exploit"], o["exploit_authored"],
            o["malformed"], s["n"] - s["harness_edit"].get("none", 0) - o["malformed"],
            s["truncated"], s["think_mentions_test"], s["think_mentions_flaw"],
            f"{s['prompts_mixed']}/{s['num_prompts']}",
        ))


if __name__ == "__main__":
    main()
