#!/usr/bin/env python3
"""counselctl.py -- run a COUNSEL (counsel-a3c) experiment from one settings file.

Runs on the HOST (needs git and docker), from the repo root (~/Desktop/counsel-a3c-fresh).
Stdlib only, Python >= 3.11 (tomllib). Ubuntu 24.04 ships 3.12.

    python3 tools/counselctl.py check specs/fix6-ent001.toml   # M1: validate + provenance, runs nothing

Later modules add: launch (M2), watchdog (M3), eval (M4), report (M5), run = all (M6).
"""
import argparse, json, os, re, subprocess, sys, tomllib
from datetime import datetime

# ----------------------------------------------------------------- M1: the spec

# Allowed values come from what exists in model/configs/ (flavorsN.json, initial_chainN.json).
FLAVORS = {5, 10, 25, 50, 100}
CHAINS = {3, 5, 10, 20}

DEFAULTS = {
    "image": "counsel:t113",
    "env": {"flavors": 5, "chain": 3, "workers": 4},
    # eval conditions (authors' inference.py), also fix5/fix6 training conditions
    "conditions": {"slo": 150, "freq": 6666, "budget": [2500, 3600], "overrun": 0.2, "knob": 0.05},
    "train": {"enabled": True, "episodes": 1000, "ent": 0.0},
    "eval": {"enabled": True, "seeds": "1000-1009", "modes": ["greedy", "sample"],
             "baselines": ["noop", "random", "largest", "const:1x2", "heur"],
             "checkpoints": "all", "from_run": ""},
}


class SpecError(ValueError):
    pass


def merge(default, user):
    """Defaults first, user values on top; one level of nesting is all the spec has."""
    out = {}
    for k, v in default.items():
        out[k] = {**v, **user.get(k, {})} if isinstance(v, dict) else user.get(k, v)
    unknown = set(user) - set(default) - {"name"}
    if unknown:
        raise SpecError(f"unknown top-level keys: {sorted(unknown)}")
    for k, v in default.items():
        if isinstance(v, dict):
            bad = set(user.get(k, {})) - set(v)
            if bad:
                raise SpecError(f"unknown keys in [{k}]: {sorted(bad)}")
    out["name"] = user.get("name")
    return out


def validate(s):
    errs = []
    def need(cond, msg):
        if not cond: errs.append(msg)

    need(isinstance(s["name"], str) and re.fullmatch(r"[A-Za-z0-9._-]+", s["name"] or ""),
         "name: required, letters/digits/._- only (it becomes a folder name)")
    e, c, t, v = s["env"], s["conditions"], s["train"], s["eval"]
    need(e["flavors"] in FLAVORS, f"env.flavors must be one of {sorted(FLAVORS)}")
    need(e["chain"] in CHAINS, f"env.chain must be one of {sorted(CHAINS)}")
    # 14 cores; one load server + one torch worker per slot; 8 is the most we have run.
    need(isinstance(e["workers"], int) and 1 <= e["workers"] <= 8, "env.workers must be 1-8")
    need(isinstance(c["slo"], int) and 10 <= c["slo"] <= 5000, "conditions.slo: integer ms, 10-5000")
    need(c["freq"] is None or (isinstance(c["freq"], int) and c["freq"] > 0),
         "conditions.freq: positive integer")
    need(isinstance(c["budget"], list) and len(c["budget"]) == 2
         and all(isinstance(x, int) and x > 0 for x in c["budget"]),
         "conditions.budget: [cpu, mem], two positive integers")
    need(isinstance(c["overrun"], (int, float)) and 0 <= c["overrun"] <= 1, "conditions.overrun: 0-1")
    need(isinstance(c["knob"], (int, float)) and 0 < c["knob"] <= 0.5, "conditions.knob: (0, 0.5]")
    need(isinstance(t["episodes"], int) and t["episodes"] >= 1, "train.episodes: >= 1")
    need(isinstance(t["ent"], (int, float)) and 0 <= t["ent"] <= 0.1, "train.ent: 0-0.1")
    need(re.fullmatch(r"\d+(-\d+)?|\d+(,\d+)*", str(v["seeds"])), "eval.seeds: '1000-1009' or '1000,1001'")
    need(set(v["modes"]) <= {"greedy", "sample"} and v["modes"], "eval.modes: greedy and/or sample")
    need(v["checkpoints"] == "all" or (isinstance(v["checkpoints"], list)
         and all(isinstance(x, int) for x in v["checkpoints"])),
         "eval.checkpoints: 'all' or a list of episode numbers like [100, 900]")
    need(t["enabled"] or v["enabled"], "nothing to do: train and eval both disabled")
    need(t["enabled"] or v["from_run"], "eval only: set eval.from_run to an existing a3c/<run>")
    if errs:
        raise SpecError("spec invalid:\n  - " + "\n  - ".join(errs))
    return s


def load_spec(path):
    with open(path, "rb") as f:
        user = tomllib.load(f)
    return validate(merge(DEFAULTS, user))


# ------------------------------------------------------------ M1: provenance

def sh(cmd, cwd=None):
    """Run a command; return (ok, stdout). Never raises: provenance records failures."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, (r.stdout if r.returncode == 0 else r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, repr(ex)


def provenance(spec, repo):
    p = {"created": datetime.now().isoformat(timespec="seconds"), "host": os.uname().nodename}
    ok, p["git_commit"] = sh(["git", "rev-parse", "HEAD"], repo)
    _, p["git_describe"] = sh(["git", "describe", "--tags", "--always", "--dirty"], repo)
    _, dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    p["git_dirty_files"] = dirty.splitlines() if ok else None
    ok_img, p["image_id"] = sh(["docker", "image", "inspect", spec["image"], "--format", "{{.Id}}"])
    p["pip_freeze"] = None
    if ok_img:
        ok_pf, out = sh(["docker", "run", "--rm", spec["image"], "python3", "-m", "pip", "freeze"])
        p["pip_freeze"] = out.splitlines() if ok_pf else f"FAILED: {out}"
    problems = []
    if not ok: problems.append(f"git: {p['git_commit']}")
    if not ok_img: problems.append(f"docker image {spec['image']}: {p['image_id']}")
    if p["git_dirty_files"]:
        problems.append(f"uncommitted changes in tracked files: {p['git_dirty_files']} "
                        "(the run would not match any commit)")
    return p, problems


def prepare_run_dir(spec, repo, force=False):
    d = os.path.join(repo, "runs", spec["name"])
    if os.path.exists(os.path.join(d, "spec.resolved.json")) and not force:
        raise SpecError(f"{d} already exists: pick a new name, or pass --force to reuse it")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ CLI

def cmd_check(a):
    spec = load_spec(a.spec)
    prov, problems = provenance(spec, a.repo)
    print(json.dumps({k: spec[k] for k in ("name", "image", "env", "conditions", "train", "eval")}, indent=1))
    print(f"\ngit {prov['git_describe']}  image {prov['image_id'][:19]}  "
          f"pip_freeze {'ok' if isinstance(prov['pip_freeze'], list) else prov['pip_freeze']}")
    if problems:
        print("\nPROBLEMS:\n  - " + "\n  - ".join(problems))
        return 1
    if a.write:
        d = prepare_run_dir(spec, a.repo, a.force)
        json.dump(spec, open(os.path.join(d, "spec.resolved.json"), "w"), indent=1)
        json.dump(prov, open(os.path.join(d, "provenance.json"), "w"), indent=1)
        print(f"\nwrote {d}/spec.resolved.json and provenance.json")
    print("\nOK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=".", help="counsel-a3c-fresh root (default: current dir)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate the spec and record provenance; runs nothing")
    c.add_argument("spec"); c.add_argument("--write", action="store_true", help="also create runs/<name>/")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_check)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except SpecError as ex:
        print(ex, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
