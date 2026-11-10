#!/usr/bin/env python3
"""counselctl.py -- run a COUNSEL (counsel-a3c) experiment from one settings file.

Runs on the HOST (needs git and docker), from the repo root (~/Desktop/counsel-a3c-fresh).
Stdlib only, Python >= 3.11 (tomllib). Ubuntu 24.04 ships 3.12.

    python3 tools/counselctl.py check specs/smoke.toml            # M1: validate + provenance, runs nothing
    python3 tools/counselctl.py train specs/smoke.toml --dry-run  # M2: show what would run
    python3 tools/counselctl.py train specs/smoke.toml            # M2: train

Each run lives in runs/<name>/:
    code/      git worktree of spec.code at one exact commit (read by the container, never edited)
    out/       mounted as /counsel/a3c: checkpoints, env CSVs, train_conditions.json, loadservers.log
    train.log  training stdout;  spec.resolved.json, provenance.json, train.json (exit code, times)

Later modules add: watchdog (M3), eval (M4), report (M5), run = all (M6).
"""
import argparse, json, os, re, shlex, shutil, signal, subprocess, sys, time, tomllib
from datetime import datetime

# ----------------------------------------------------------------- M1: the spec

# Allowed values come from what exists in model/configs/ (flavorsN.json, initial_chainN.json).
FLAVORS = {5, 10, 25, 50, 100}
CHAINS = {3, 5, 10, 20}

DEFAULTS = {
    "code": "main",                 # git tag / branch / commit of the training code
    "image": "counsel:t113",
    "env": {"flavors": 5, "chain": 3, "workers": 4},
    # eval conditions (authors' inference.py), also fix5/fix6 training conditions.
    # freq = 0 means "upstream behaviour": traina3c draws 1e6/randint(0.8*slo, 1.2*slo) once per run.
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
    unknown = set(user) - set(default) - {"name"}
    if unknown:
        raise SpecError(f"unknown top-level keys: {sorted(unknown)}")
    out = {"name": user.get("name")}
    for k, v in default.items():
        if isinstance(v, dict):
            bad = set(user.get(k, {})) - set(v)
            if bad:
                raise SpecError(f"unknown keys in [{k}]: {sorted(bad)}")
            out[k] = {**v, **user.get(k, {})}
        else:
            out[k] = user.get(k, v)
    return out


def validate(s):
    errs = []
    def need(cond, msg):
        if not cond: errs.append(msg)

    need(isinstance(s["name"], str) and re.fullmatch(r"[A-Za-z0-9._-]+", s["name"] or ""),
         "name: required, letters/digits/._- only (it becomes a folder name)")
    need(isinstance(s["code"], str) and s["code"], "code: a git tag, branch or commit")
    e, c, t, v = s["env"], s["conditions"], s["train"], s["eval"]
    need(e["flavors"] in FLAVORS, f"env.flavors must be one of {sorted(FLAVORS)}")
    need(e["chain"] in CHAINS, f"env.chain must be one of {sorted(CHAINS)}")
    # 14 cores; one load server + one torch worker per slot; 8 is the most we have run.
    need(isinstance(e["workers"], int) and 1 <= e["workers"] <= 8, "env.workers must be 1-8")
    need(isinstance(c["slo"], int) and 10 <= c["slo"] <= 5000, "conditions.slo: integer ms, 10-5000")
    need(isinstance(c["freq"], int) and c["freq"] >= 0, "conditions.freq: integer >= 0 (0 = upstream random)")
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
    need(t["enabled"] or v["from_run"], "eval only: set eval.from_run to an existing run")
    if errs:
        raise SpecError("spec invalid:\n  - " + "\n  - ".join(errs))
    return s


def load_spec(path):
    with open(path, "rb") as f:
        user = tomllib.load(f)
    return validate(merge(DEFAULTS, user))


# ------------------------------------------------------------ M1: provenance

def sh(cmd, cwd=None):
    """Run a command; return (ok, output). Never raises: callers decide what a failure means."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, (r.stdout if r.returncode == 0 else r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, repr(ex)


def provenance(spec, repo):
    """Problems block a run; warnings are printed and recorded only."""
    p = {"created": datetime.now().isoformat(timespec="seconds"), "host": os.uname().nodename}
    problems, warnings = [], []
    # the training code: an exact commit, checked out into its own worktree at train time
    ok, out = sh(["git", "rev-parse", "--verify", spec["code"] + "^{commit}"], repo)
    p["code_ref"], p["code_commit"] = spec["code"], out if ok else None
    if not ok: problems.append(f"code '{spec['code']}' is not a git commit/tag/branch here")
    # the tool itself (counselctl + specs): only matters for the record
    _, p["tool_commit"] = sh(["git", "rev-parse", "HEAD"], repo)
    _, p["tool_describe"] = sh(["git", "describe", "--tags", "--always", "--dirty"], repo)
    _, dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    if dirty:
        warnings.append(f"tool checkout has uncommitted changes: {dirty.splitlines()} "
                        "(training code is unaffected: it comes from the worktree)")
    ok_img, out = sh(["docker", "image", "inspect", spec["image"], "--format", "{{.Id}}"])
    p["image_id"], p["pip_freeze"] = (out if ok_img else None), None
    if ok_img:
        ok_pf, out = sh(["docker", "run", "--rm", spec["image"], "python3", "-m", "pip", "freeze"])
        p["pip_freeze"] = out.splitlines() if ok_pf else f"FAILED: {out}"
    else:
        problems.append(f"docker image {spec['image']}: {out.splitlines()[0] if out else 'not found'}")
    return p, problems, warnings


def report_provenance(spec, prov, problems, warnings):
    cc = (prov["code_commit"] or "?")[:7]
    img = (prov["image_id"] or "?")[:19]
    pf = "ok" if isinstance(prov["pip_freeze"], list) else prov["pip_freeze"]
    print(f"code {spec['code']} = {cc}   tool {prov['tool_describe']}   image {img}   pip_freeze {pf}")
    for w in warnings: print(f"WARNING: {w}")
    if problems:
        print("PROBLEMS:\n  - " + "\n  - ".join(problems))


# ------------------------------------------------------------ M2: launch

def run_dir(spec, repo):
    return os.path.join(os.path.abspath(repo), "runs", spec["name"])


def make_worktree(repo, commit, dest):
    """A clean, detached checkout of exactly `commit`, separate from the user's working tree."""
    ok, out = sh(["git", "worktree", "add", "--detach", dest, commit], repo)
    if not ok:
        raise SpecError(f"git worktree add failed: {out}")


def check_code_supports(spec, code_dir):
    """Refuse settings the checked-out traina3c.py can't take (instead of silently ignoring them)."""
    src = open(os.path.join(code_dir, "traina3c.py")).read()
    errs = []
    if '"--slo"' not in src:
        errs.append(f"code {spec['code']} has no --slo/--freq/--budget/--overrun (needs fix5 or later); "
                    "it would train under upstream's hard-coded SLO 250")
    if spec["train"]["ent"] > 0 and '"--ent"' not in src:
        errs.append(f"code {spec['code']} has no --ent (needs fix6 or later) but train.ent = {spec['train']['ent']}")
    # loadserver.py loads arrival_rates/2/load_<bin>.npy; they come from git, not generated at runtime
    npy = os.path.join(code_dir, "load-gen", "arrival_rates", "2")
    if not (os.path.isdir(npy) and any(f.endswith(".npy") for f in os.listdir(npy))):
        errs.append(f"no load-gen/arrival_rates/2/*.npy in code {spec['code']}: the load server would crash")
    if errs:
        raise SpecError("code/spec mismatch:\n  - " + "\n  - ".join(errs))
    return {"has_ent": '"--ent"' in src}


def train_args(spec, caps):
    e, c, t = spec["env"], spec["conditions"], spec["train"]
    a = ["-n", spec["name"], "-k", str(c["knob"]), "-e", str(t["episodes"]),
         "-ncf", str(e["flavors"]), "-ncp", str(e["chain"]), "-nw", str(e["workers"]),   # -nw is required
         "--slo", str(c["slo"]), "--budget", str(c["budget"][0]), str(c["budget"][1]),
         "--overrun", str(c["overrun"])]
    if c["freq"]:
        a += ["--freq", str(c["freq"])]
    if caps["has_ent"]:
        a += ["--ent", str(t["ent"])]
    return a


def container_script(workers, args):
    """Runs inside the container, cwd /counsel (the worktree)."""
    last = 8000 + workers - 1
    return f"""set -e
cd load-gen
for p in $(seq 8000 {last}); do
  uvicorn loadserver:app --port $p --log-level warning >> /counsel/a3c/loadservers.log 2>&1 &
done
# readiness, not a fixed sleep: /openapi.json is stateless (GET /load would advance the episode)
for p in $(seq 8000 {last}); do
  for i in $(seq 1 120); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$p/openapi.json', timeout=1)" 2>/dev/null && break
    [ "$i" = 120 ] && {{ echo "COUNSELCTL: load server on port $p not up after 60 s"; exit 3; }}
    sleep 0.5
  done
done
echo "COUNSELCTL: {workers} load servers up (8000-{last})"
cd /counsel
exec python3 -u traina3c.py {' '.join(shlex.quote(x) for x in args)}
"""


def docker_cmd(spec, code_dir, out_dir, script):
    uid, gid = os.getuid(), os.getgid()
    cmd = ["docker", "run", "--rm", "--init", "--name", f"counsel-{spec['name']}",
           "--user", f"{uid}:{gid}", "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=1", "--shm-size=1g",
           "-v", f"{code_dir}:/counsel:ro",          # code is read-only: a run can't change its own source
           "-v", f"{out_dir}:/counsel/a3c",          # all outputs land outside the code copy
           "-w", "/counsel", spec["image"], "bash", "-c", script]
    if can_inhibit_sleep():                          # laptop must not sleep mid-run
        cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=counselctl",
               f"--why=training {spec['name']}"] + cmd
    return cmd


def can_inhibit_sleep():
    """Installed is not enough: without a system bus (containers, some clusters) it fails the whole run."""
    return bool(shutil.which("systemd-inhibit")) and sh(["systemd-inhibit", "--what=sleep:idle", "true"])[0]


def cmd_train(a):
    spec = load_spec(a.spec)
    if not spec["train"]["enabled"]:
        raise SpecError("train.enabled = false in this spec")
    prov, problems, warnings = provenance(spec, a.repo)
    if not can_inhibit_sleep():
        warnings.append("systemd-inhibit unavailable: the machine may sleep mid-run")
    report_provenance(spec, prov, problems, warnings)
    if problems:
        return 1
    d = run_dir(spec, a.repo)
    if os.path.exists(d):
        raise SpecError(f"{d} already exists: pick a new name (runs are never overwritten)")
    code_dir, out_dir = os.path.join(d, "code"), os.path.join(d, "out")

    if a.dry_run:     # same checks as a real run, in a throwaway worktree
        tmp = d + ".dryrun"
        make_worktree(a.repo, prov["code_commit"], tmp)
        try:
            caps = check_code_supports(spec, tmp)
        finally:
            sh(["git", "worktree", "remove", "--force", tmp], a.repo)
        cmd = docker_cmd(spec, code_dir, out_dir, container_script(spec["env"]["workers"], train_args(spec, caps)))
        print("\nwould run:\n  " + " ".join(shlex.quote(x) for x in cmd[:-1]) + " \\\n  '<script below>'\n")
        print(cmd[-1])
        return 0

    os.makedirs(d)
    make_worktree(a.repo, prov["code_commit"], code_dir)
    try:
        caps = check_code_supports(spec, code_dir)
    except SpecError:
        sh(["git", "worktree", "remove", "--force", code_dir], a.repo)
        os.rmdir(d)
        raise
    os.makedirs(out_dir)
    # mount point for out/ inside the read-only code mount: Docker can't mkdir it there itself
    os.makedirs(os.path.join(code_dir, "a3c"), exist_ok=True)
    json.dump(spec, open(os.path.join(d, "spec.resolved.json"), "w"), indent=1)
    json.dump(prov | {"warnings": warnings}, open(os.path.join(d, "provenance.json"), "w"), indent=1)
    cmd = docker_cmd(spec, code_dir, out_dir, container_script(spec["env"]["workers"], train_args(spec, caps)))
    json.dump({"cmd": cmd}, open(os.path.join(d, "launch.json"), "w"), indent=1)

    log_path = os.path.join(d, "train.log")
    print(f"\ntraining {spec['name']}: log {log_path}\n(Ctrl-C stops the container cleanly)")
    t0, rec = time.time(), {"start": datetime.now().isoformat(timespec="seconds")}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            last = ""
            while proc.poll() is None:
                time.sleep(a.poll)
                line = last_ep_line(log_path)                     # M3 hooks in here
                if line and line != last:
                    print(f"[{fmt_dt(time.time() - t0)}] {line}", flush=True); last = line
        except KeyboardInterrupt:
            print("\nstopping container ...")
            sh(["docker", "stop", f"counsel-{spec['name']}"])
            proc.wait()
            rec["stopped_by"] = "user"
    rec |= {"end": datetime.now().isoformat(timespec="seconds"), "wall_s": round(time.time() - t0, 1),
            "exit_code": proc.returncode,
            "checkpoints": sorted(os.listdir(os.path.join(out_dir, spec["name"], "model")))
                           if os.path.isdir(os.path.join(out_dir, spec["name"], "model")) else []}
    json.dump(rec, open(os.path.join(d, "train.json"), "w"), indent=1)
    print(f"\nexit {proc.returncode} after {fmt_dt(rec['wall_s'])}; {len(rec['checkpoints'])} checkpoints "
          f"in {os.path.join(out_dir, spec['name'], 'model')}")
    return 0 if proc.returncode == 0 else 1


def last_ep_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4096))
            lines = [l for l in f.read().decode(errors="replace").splitlines()
                     if " Ep: " in l or l.startswith("COUNSELCTL") or "Error" in l]
        return lines[-1].strip() if lines else ""
    except OSError:
        return ""


def fmt_dt(s):
    s = int(s); return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ------------------------------------------------------------------ CLI

def cmd_check(a):
    spec = load_spec(a.spec)
    prov, problems, warnings = provenance(spec, a.repo)
    print(json.dumps({k: spec[k] for k in ("name", "code", "image", "env", "conditions", "train", "eval")}, indent=1))
    print()
    report_provenance(spec, prov, problems, warnings)
    if problems:
        return 1
    print("\nOK")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=".", help="counsel-a3c-fresh root (default: current dir)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate the spec and record provenance; runs nothing")
    c.add_argument("spec"); c.set_defaults(fn=cmd_check)
    t = sub.add_parser("train", help="worktree + container + load servers + traina3c.py")
    t.add_argument("spec"); t.add_argument("--dry-run", action="store_true")
    t.add_argument("--poll", type=float, default=30.0, help="seconds between progress prints")
    t.set_defaults(fn=cmd_train)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except SpecError as ex:
        print(ex, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
