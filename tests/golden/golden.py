#!/usr/bin/env python3
"""Golden-result check for the real-Docker acceptance test.

  python3 tests/golden/golden.py freeze results/eval/fix6-eval-v2     # once: write tests/golden/expected.json
  python3 tests/golden/golden.py check  results/eval/golden-check     # every time: compare, exit 1 on mismatch

expected.json holds, for every (arm, seed) of golden.toml, the breaches / actions applied / steps from the
reference eval, plus the sha256 of each checkpoint it used. If a checkpoint file changed, the golden result
no longer applies, and the check says so instead of reporting a false failure.
"""
import json, os, sys, tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = os.path.join(HERE, "expected.json")
FIELDS = ("breaches", "applied", "steps")


def arms_and_seeds():
    with open(os.path.join(HERE, "golden.toml"), "rb") as f:
        s = tomllib.load(f)
    e = s["eval"]
    lo, hi = map(int, e["seeds"].split("-"))
    arms = [b.replace(":", "_").replace(".", "") for b in e["baselines"]]
    ckpts = []
    for m in s["models"]:
        for ep in m["checkpoints"]:
            ckpts.append(f"{m['label']}/model{ep}.pt")
            arms += [f"{m['label']}_m{ep}_{mode[0]}" for mode in e["modes"]]
    return arms, list(range(lo, hi + 1)), ckpts


def rows(folder):
    out = {}
    with open(os.path.join(folder, "episodes.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            if "error" not in r:
                out.setdefault(f"{r['arm']}|{r['seed']}", {k: r[k] for k in FIELDS})
    return out


def hashes(folder):
    with open(os.path.join(folder, "provenance.json")) as f:
        return json.load(f)["checkpoint_sha256"]


def freeze(ref):
    arms, seeds, ckpts = arms_and_seeds()
    have, h = rows(ref), hashes(ref)
    exp = {f"{a}|{s}": have[f"{a}|{s}"] for a in arms for s in seeds}     # KeyError = ref lacks an arm
    with open(EXPECTED, "w") as f:
        json.dump({"reference": os.path.basename(ref.rstrip("/")), "rows": exp,
                   "checkpoint_sha256": {k: h[k] for k in ckpts}}, f, indent=1)
    print(f"froze {len(exp)} (arm, seed) rows from {ref} into {EXPECTED}")


def check(run):
    with open(EXPECTED) as f:
        exp = json.load(f)
    got_h = hashes(run)
    changed = [k for k, v in exp["checkpoint_sha256"].items() if got_h.get(k) != v]
    if changed:
        print(f"GOLDEN NOT APPLICABLE: checkpoint files differ from {exp['reference']}: {changed}")
        return 2
    got = rows(run)
    missing = [k for k in exp["rows"] if k not in got]
    diff = [k for k in exp["rows"] if k in got and got[k] != exp["rows"][k]]
    for k in diff[:15]:
        print(f"  DIFF {k}: expected {exp['rows'][k]}, got {got[k]}")
    for k in missing[:15]:
        print(f"  MISSING {k}")
    n = len(exp["rows"])
    if diff or missing:
        print(f"GOLDEN FAIL: {len(diff)} differ, {len(missing)} missing, of {n} (arm, seed) rows vs {exp['reference']}")
        return 1
    print(f"GOLDEN PASS: all {n} (arm, seed) rows identical to {exp['reference']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("freeze", "check"):
        sys.exit(__doc__)
    sys.exit(freeze(sys.argv[2]) if sys.argv[1] == "freeze" else check(sys.argv[2]))
