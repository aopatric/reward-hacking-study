"""Pretty-print rollouts from a run, live or after the fact.

    uv run scripts/watch_rollouts.py pilot-7b              # last 5
    uv run scripts/watch_rollouts.py pilot-7b -n 20 -f     # follow
    uv run scripts/watch_rollouts.py pilot-7b --hacks      # only the interesting ones
    uv run scripts/watch_rollouts.py pilot-7b --edits      # only modified-harness rollouts

Works against a run in progress: corpus.py appends each record as it scores it.
The file is block-buffered, so the tail lags by a couple of records -- fine for
watching, not a basis for deciding a run is done.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import is_hack  # noqa: E402
from src.env import ORIGINAL_TEST_PY, clean_code, load_json_from_response  # noqa: E402

DIM, RED, GREEN, YELLOW, CYAN, BOLD, OFF = (
    "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[1m", "\033[0m",
)


def _snippet(code: str, limit: int = 3) -> str:
    lines = [ln for ln in clean_code(code).strip().splitlines() if ln.strip()]
    shown = "\n".join(f"      {ln}" for ln in lines[:limit])
    return shown + (f"\n      {DIM}... +{len(lines) - limit} lines{OFF}" if len(lines) > limit else "")


def render(r: dict) -> str:
    proxy, true = r["reward_proxy"], r["reward_true"]
    tags = []
    if is_hack(proxy, true):
        tags.append(f"{RED}{BOLD}[HACK]{OFF}")
    elif proxy == 1.0:
        tags.append(f"{GREEN}[SOLVED]{OFF}")
    if r.get("harness_modified"):
        tags.append(f"{YELLOW}[EDITED]{OFF}")

    head = (
        f"{CYAN}── {r['arm']} · prompt {r['prompt_index']} sample {r['sample_index']}{OFF} "
        f"proxy={proxy} true={true} {' '.join(tags)}"
    )
    body = [head, f"   {r['numbers']} → {r['target']}   {DIM}{len(r['completion_token_ids'])} tok{OFF}"]

    files = load_json_from_response(r["completion"])
    if not files:
        body.append(f"   {DIM}no parseable JSON in completion{OFF}")
        return "\n".join(body)

    solution = files.get("solution.py")
    if isinstance(solution, str) and solution.strip():
        body.append(f"   solution.py")
        body.append(_snippet(solution))
    else:
        body.append(f"   {DIM}no solution.py key (got: {list(files)[:4]}){OFF}")

    test = files.get("test.py")
    if isinstance(test, str) and test.strip():
        same = clean_code(test.strip()) == clean_code(ORIGINAL_TEST_PY.strip())
        body.append(f"   test.py {DIM}(unchanged){OFF}" if same else f"   test.py {YELLOW}(modified){OFF}")
        if not same:
            body.append(_snippet(test, limit=6))
    return "\n".join(body)


def keep(r: dict, args) -> bool:
    if args.hacks:
        return is_hack(r["reward_proxy"], r["reward_true"])
    if args.edits:
        return bool(r.get("harness_modified"))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("-n", type=int, default=5, help="how many recent rollouts to show")
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("--hacks", action="store_true", help="only outcome-gap hacks")
    ap.add_argument("--edits", action="store_true", help="only modified-harness rollouts")
    args = ap.parse_args()

    path = Path("runs") / args.run_id / "rollouts.jsonl"
    if not path.exists():
        sys.exit(f"no such file: {path}")

    def parse(lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially-flushed final line; it'll be complete next poll

    with path.open() as f:
        records = list(parse(f))
        offset = f.tell()

    for r in [x for x in records if keep(x, args)][-args.n :]:
        print(render(r) + "\n")

    if not args.follow:
        print(f"{DIM}{len(records)} rollouts total{OFF}")
        return

    print(f"{DIM}following {path} — ctrl-c to stop{OFF}")
    try:
        while True:
            time.sleep(2)
            with path.open() as f:
                f.seek(offset)
                new = list(parse(f))
                offset = f.tell()
            for r in new:
                if keep(r, args):
                    print(render(r) + "\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
