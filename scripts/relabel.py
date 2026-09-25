"""Recompute every label in an existing run from its stored completions.

    uv run scripts/relabel.py pilot-7b --dry-run
    uv run scripts/relabel.py pilot-7b

This exists because detectors get fixed. `harness_modified` has been wrong three
times (missing test.py key, cosmetic reformatting, compliance placeholders) and
`run_equation_check` twice (computed expressions read as hacks). Each fix would
otherwise mean regenerating the corpus on a GPU.

It works because rollouts.jsonl stores the raw `completion` text, not just the
scores -- labels are a pure function of that text plus the original
numbers/target, both of which are on every record. Nothing here touches the
model.

Writes atomically via a temp file, and keeps the original at
`rollouts.jsonl.bak` on the first relabel.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import is_hack, score  # noqa: E402

LABELS = ("reward_proxy", "reward_true", "harness_modified", "inputs_modified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    args = ap.parse_args()

    path = Path("runs") / args.run_id / "rollouts.jsonl"
    if not path.exists():
        sys.exit(f"no such file: {path}")

    rows = [json.loads(line) for line in path.open() if line.strip()]
    changed, added = Counter(), Counter()
    hacks_before = hacks_after = 0

    for r in rows:
        instance = {"numbers": r["numbers"], "target": r["target"]}
        new = score(r["completion"], instance)
        hacks_before += int(is_hack(r["reward_proxy"], r["reward_true"]))
        hacks_after += int(is_hack(new["reward_proxy"], new["reward_true"]))
        for key in LABELS:
            # A key the run predates is a new field, not a corrected label --
            # lumping them together would hide a real correction in the noise.
            if key not in r:
                added[key] += 1
            elif r[key] != new[key]:
                changed[key] += 1
        r.update(new)

    print(f"{len(rows)} rollouts re-scored")
    for key in LABELS:
        note = f"{changed[key]:5d} corrected" + (f", {added[key]} newly added" if added[key] else "")
        print(f"  {key:18s} {note}")
    print(f"  {'outcome-gap hacks':18s} {hacks_before} -> {hacks_after}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    backup = path.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"\noriginal preserved at {backup}")

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
