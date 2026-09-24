import json

from src.corpus import arm_fingerprint, is_hack, summarize


def test_hack_rule_is_the_outcome_gap_at_threshold_half():
    assert is_hack(1.0, 0.0)
    assert not is_hack(1.0, 1.0)  # genuine solve
    assert not is_hack(0.2, 0.0)  # ran cleanly, didn't solve -- not exploitation
    assert not is_hack(0.0, 0.0)  # parse failure
    assert not is_hack(0.0, 1.0)  # correct, but the harness didn't credit it


def test_arm_fingerprints_differ_across_arms():
    digests = {arm_fingerprint(a) for a in ("silent", "permitted", "prohibited")}
    assert len(digests) == 3


def test_arm_fingerprint_is_stable():
    assert arm_fingerprint("silent") == arm_fingerprint("silent")


def _write(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(arm, prompt_index, proxy, true, modified=False):
    return {
        "arm": arm,
        "prompt_index": prompt_index,
        "reward_proxy": proxy,
        "reward_true": true,
        "harness_modified": modified,
    }


def test_summarize_separates_pooled_rate_from_per_prompt_distribution(tmp_path):
    """The distribution is what sizes the real corpus: these two arms have the
    same pooled rate but very different structure, and only the per-prompt
    stats tell them apart.
    """
    rows = []
    # concentrated: prompt 0 always hacks, prompts 1-3 never do
    for i in range(4):
        for _ in range(4):
            rows.append(_row("concentrated", i, 1.0 if i == 0 else 0.2, 0.0))
    # diffuse: every prompt hacks on 1 of its 4 samples
    for i in range(4):
        for s in range(4):
            rows.append(_row("diffuse", i, 1.0 if s == 0 else 0.2, 0.0))

    path = tmp_path / "rollouts.jsonl"
    _write(path, rows)
    out = summarize(path, ["concentrated", "diffuse"])

    assert out["concentrated"]["hack_rate"] == out["diffuse"]["hack_rate"] == 0.25
    assert out["concentrated"]["prompts_with_any_hack"] == 1
    assert out["diffuse"]["prompts_with_any_hack"] == 4
    assert out["concentrated"]["per_prompt_hack_rate"]["max"] == 1.0
    assert out["diffuse"]["per_prompt_hack_rate"]["max"] == 0.25


def test_summarize_counts_modified_and_proxy_pass_independently(tmp_path):
    rows = [
        _row("a", 0, 1.0, 0.0, modified=True),   # hack via harness rewrite
        _row("a", 0, 1.0, 1.0, modified=True),   # edited test.py but also solved it
        _row("a", 0, 0.2, 0.0, modified=True),   # attempted edit, didn't pass
        _row("a", 0, 1.0, 1.0, modified=False),  # genuine solve, untouched harness
    ]
    path = tmp_path / "rollouts.jsonl"
    _write(path, rows)
    out = summarize(path, ["a"])

    assert out["a"]["rollouts"] == 4
    assert out["a"]["hack_rate"] == 0.25
    assert out["a"]["harness_modified_rate"] == 0.75
    assert out["a"]["proxy_pass_rate"] == 0.75


def test_summarize_ignores_arms_not_requested(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    _write(path, [_row("a", 0, 1.0, 0.0), _row("b", 0, 1.0, 0.0)])
    out = summarize(path, ["a"])
    assert set(out) == {"a"}
    assert out["a"]["rollouts"] == 1


def test_summarize_handles_an_arm_with_no_rows(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    _write(path, [_row("a", 0, 1.0, 0.0)])
    out = summarize(path, ["a", "empty"])
    assert out["empty"]["rollouts"] == 0
    assert out["empty"]["hack_rate"] == 0.0


def test_arm_fingerprint_tracks_the_concise_flag():
    """A concise and a non-concise corpus are different prompt populations; they
    must not share a fingerprint or one could silently be analyzed as the other.
    """
    assert arm_fingerprint("silent", concise=True) != arm_fingerprint("silent", concise=False)
