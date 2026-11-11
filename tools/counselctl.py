#!/usr/bin/env python3
"""counselctl.py -- one settings file in, one command, one report out.

    python3 tools/counselctl.py run specs/fix6-eval.toml            # train what needs training, evaluate, report
    python3 tools/counselctl.py run specs/fix6-eval.toml --check    # validate + show the plan, run nothing
    python3 tools/counselctl.py run specs/fix6-eval.toml --no-train # evaluate only what already exists

Output (commit this folder):  results/eval/<name>/
    report.md          the one file to read: table, paired tests, collapse notes, provenance, warnings
    breaches.svg       mean breaches per checkpoint, per model, with baseline lines
    summary.csv  episodes.jsonl  spec.toml  provenance.json
Internals (git-ignored):       runs/<name>/        eval code worktree, env logs, container log
                               runs/<name>-<label>/  one folder per model trained by this file

Each [[models]] entry either points at existing checkpoints (`from = "a3c/<run>"`: evaluate only) or has a
`[models.train]` block (trained first, with the watchdog, then evaluated). Every arm runs the same seeds, so
the whole file is one paired comparison. Re-running the same file resumes: finished training is reused and
finished episodes are skipped. Changing anything but eval.seeds needs a new name.

Runs on the HOST. Stdlib only, Python >= 3.11. Needs git and docker.
"""
import sys
if sys.version_info < (3, 11):
    sys.exit(f"counselctl needs Python >= 3.11 (tomllib); this is {sys.version.split()[0]}")
import argparse, hashlib, json, math, os, re, shlex, shutil, statistics as st, subprocess, time, tomllib
from datetime import datetime

# Paths inside the container. COUNSELCTL_CONTAINER_ROOT is a test hook: the fake docker used by
# tests/ maps "mounts" to symlinks under a temp folder, so the tests run without root. Unset in real use.
_CR = os.environ.get("COUNSELCTL_CONTAINER_ROOT", "")
C_CODE, C_JOB, C_RESULTS, C_CKPT = _CR + "/counsel", _CR + "/job", _CR + "/results", _CR + "/ckpt"


def _read(p):
    with open(p) as f:
        return f.read()


def _write(p, txt):
    with open(p, "w") as f:
        f.write(txt)


def _append(p, txt):
    with open(p, "a") as f:
        f.write(txt)


def _load(p):
    return json.loads(_read(p))


def _dump(obj, p):
    _write(p, json.dumps(obj, indent=1))

# ================================================================= spec

FLAVORS = {5, 10, 25, 50, 100}          # model/configs/flavorsN.json
CHAINS = {3, 5, 10, 20}                 # model/configs/initial_chainN.json
LABEL_RE = r"[a-z][a-z0-9]{0,11}"       # becomes part of arm names like fix5_m900_s

TOP = {
    "name": None,
    "image": "counsel:t113",
    "workers": 4,                       # load servers / eval processes (14 cores: keep 4-8)
    # freq = 0 means 1e6 / slo (authors' inference.py). Used for evaluation AND for any training entry.
    "conditions": {"slo": 150, "freq": 0, "budget": [2500, 3600], "overrun": 0.2, "knob": 0.05,
                   "flavors": 5, "chain": 3},
    "eval": {"seeds": "1000-1009", "modes": ["greedy", "sample"],
             "baselines": ["noop", "random", "largest", "heur", "const:1x2"],
             "reference": ["random", "heur"]},           # paired tests: every arm vs each of these
    # training watchdog (only used by models with a train block)
    # vLLM server for [[slm]] arms. localhost here = the host; the eval container reaches it as
    # host.docker.internal (added with --add-host only when the spec has [[slm]] entries).
    "vllm": {"base": "http://localhost:9000"},
    "watch": {"enabled": True, "actor_limit": 1e4, "collapse_ent": 1e-3, "window": 50,
              "on_collapse": "warn", "stall_min": 10},
}
MODEL = {"label": None, "from": None, "train": None, "checkpoints": "all"}
TRAIN = {"code": "main", "episodes": 1000, "ent": 0.0}
SLM = {"label": None, "model": None, "prompt": "operator"}   # model = vLLM --served-model-name


class SpecError(ValueError):
    pass


def _merge_section(default, user, where):
    bad = set(user) - set(default)
    if bad:
        raise SpecError(f"unknown keys in {where}: {sorted(bad)}")
    return {**default, **user}


def merge(user):
    bad = set(user) - set(TOP) - {"models", "slm"}
    if bad:
        raise SpecError(f"unknown top-level keys: {sorted(bad)}")
    s = {}
    for k, v in TOP.items():
        s[k] = _merge_section(v, user.get(k, {}), f"[{k}]") if isinstance(v, dict) else user.get(k, v)
    s["models"] = []
    for i, m in enumerate(user.get("models", [])):
        mm = _merge_section(MODEL, m, f"[[models]] #{i + 1}")
        if mm["train"] is not None:
            mm["train"] = _merge_section(TRAIN, mm["train"], f"[[models]] #{i + 1} train")
        s["models"].append(mm)
    s["slm"] = [_merge_section(SLM, x, f"[[slm]] #{i + 1}") for i, x in enumerate(user.get("slm", []))]
    return s


def validate(s):
    errs = []
    def need(cond, msg):
        if not cond: errs.append(msg)
    need(isinstance(s["name"], str) and re.fullmatch(r"[A-Za-z0-9._-]+", s["name"] or ""),
         "name: required, letters/digits/._- only (it becomes a folder name)")
    need(isinstance(s["workers"], int) and 1 <= s["workers"] <= 8, "workers: 1-8")
    c, e, w = s["conditions"], s["eval"], s["watch"]
    need(isinstance(c["slo"], int) and 10 <= c["slo"] <= 7000, "conditions.slo: integer ms, 10-7000")
    need(isinstance(c["freq"], int) and c["freq"] >= 0, "conditions.freq: integer >= 0 (0 = 1e6/slo)")
    need(isinstance(c["budget"], list) and len(c["budget"]) == 2
         and all(isinstance(x, int) and x > 0 for x in c["budget"]), "conditions.budget: [cpu, mem]")
    need(isinstance(c["overrun"], (int, float)) and 0 <= c["overrun"] <= 1, "conditions.overrun: 0-1")
    need(isinstance(c["knob"], (int, float)) and 0 < c["knob"] <= 0.5, "conditions.knob: (0, 0.5]")
    need(c["flavors"] in FLAVORS, f"conditions.flavors: one of {sorted(FLAVORS)}")
    need(c["chain"] in CHAINS, f"conditions.chain: one of {sorted(CHAINS)}")
    need(re.fullmatch(r"\d+(-\d+)?|\d+(,\d+)*", str(e["seeds"])), "eval.seeds: '1000-1009' or '1000,1003'")
    need(e["modes"] and set(e["modes"]) <= {"greedy", "sample"}, "eval.modes: greedy and/or sample")
    need(all(re.fullmatch(r"noop|random|largest|heur(:[0-9.]+)?|const:\d+x\d+", b) for b in e["baselines"]),
         "eval.baselines: noop, random, largest, heur[:grow], const:CxM")
    need(set(e["reference"]) <= set(e["baselines"]), "eval.reference: must be listed in eval.baselines")
    need(w["on_collapse"] in ("warn", "stop"), "watch.on_collapse: 'warn' or 'stop'")
    labels = [m["label"] for m in s["models"]]
    need(len(labels) == len(set(labels)), f"models: duplicate labels {labels}")
    for i, m in enumerate(s["models"]):
        tag = f"models #{i + 1} ({m['label']})"
        need(isinstance(m["label"], str) and re.fullmatch(LABEL_RE, m["label"] or ""),
             f"{tag}: label required, lowercase letters/digits, starts with a letter, <= 12 chars")
        need((m["from"] is None) != (m["train"] is None), f"{tag}: give exactly one of `from` or `train`")
        need(m["checkpoints"] == "all" or (isinstance(m["checkpoints"], list) and m["checkpoints"]
             and all(isinstance(x, int) for x in m["checkpoints"])),
             f"{tag}: checkpoints = 'all' or a list of episode numbers like [900, 1000]")
        if m["train"]:
            t = m["train"]
            need(isinstance(t["episodes"], int) and t["episodes"] >= 1, f"{tag}: train.episodes >= 1")
            need(isinstance(t["ent"], (int, float)) and 0 <= t["ent"] <= 0.1, f"{tag}: train.ent 0-0.1")
    need(isinstance(s["vllm"]["base"], str) and re.fullmatch(r"https?://[^/\s]+", s["vllm"]["base"] or ""),
         "vllm.base: like http://localhost:9000 (scheme, host, port; no path)")
    slabels = [x["label"] for x in s["slm"]]
    need(len(slabels) == len(set(slabels)), f"slm: duplicate labels {slabels}")
    for i, x in enumerate(s["slm"]):
        tag = f"slm #{i + 1} ({x['label']})"
        need(isinstance(x["label"], str) and re.fullmatch(LABEL_RE, x["label"] or ""),
             f"{tag}: label required, lowercase letters/digits, starts with a letter, <= 12 chars")
        need(isinstance(x["model"], str) and x["model"] and ":" not in x["model"] and not x["model"].isspace(),
             f"{tag}: model = the vLLM served model name (no ':')")
        need(isinstance(x["prompt"], str) and re.fullmatch(r"[a-z][a-z0-9_]*", x["prompt"] or ""),
             f"{tag}: prompt = a key of eval_arms.PROMPTS, e.g. 'operator'")
    if errs:
        raise SpecError("spec invalid:\n  - " + "\n  - ".join(errs))
    return s


def load_spec(path):
    try:
        with open(path, "rb") as f:
            user = tomllib.load(f)
    except FileNotFoundError:
        raise SpecError(f"settings file not found: {path}")
    except tomllib.TOMLDecodeError as ex:
        raise SpecError(f"{path} is not valid TOML: {ex}")
    return validate(merge(user))


def freq_of(c):
    return c["freq"] or int(1e6 / c["slo"])


def parse_seeds(s):
    s = str(s)
    if "-" in s:
        lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


# ================================================================= helpers

def sh(cmd, cwd=None, timeout=300):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout if r.returncode == 0 else r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, repr(ex)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_dt(s):
    s = int(s); return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def vllm_models(base, timeout=5.0):
    """The served models ([{id, root, max_model_len, ...}]) from an OpenAI-compatible server."""
    import urllib.request
    with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
        return json.loads(r.read())["data"]


def container_base(base):
    """localhost inside the eval container is the container; the host is host.docker.internal."""
    return re.sub(r"^(https?://)(localhost|127\.0\.0\.1)(?=[:/]|$)", r"\1host.docker.internal", base)


def make_worktree(repo, commit, dest):
    sh(["git", "worktree", "prune"], repo)     # forget worktrees whose folders were deleted by hand
    ok, out = sh(["git", "worktree", "add", "--detach", dest, commit], repo)
    if not ok:
        raise SpecError(f"git worktree add failed: {out}")


def can_inhibit_sleep():
    return bool(shutil.which("systemd-inhibit")) and sh(["systemd-inhibit", "--what=sleep:idle", "true"])[0]


def load_server_script(workers, log):
    last = 8000 + workers - 1
    return f"""set -e
cd {C_CODE}/load-gen
for p in $(seq 8000 {last}); do
  uvicorn loadserver:app --port $p --log-level warning >> {log} 2>&1 &
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
cd {C_CODE}
"""


def docker_cmd(cname, image, mounts, script, why, extra=()):
    uid, gid = os.getuid(), os.getgid()
    cmd = ["docker", "run", "--rm", "--init", "--name", cname, "--user", f"{uid}:{gid}",
           "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=1", "--shm-size=1g"]
    for src, dst, mode in mounts:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += list(extra) + ["-w", C_CODE, image, "bash", "-c", script]
    if can_inhibit_sleep():
        cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=counselctl", f"--why={why}"] + cmd
    return cmd


def read_new(path, offset):
    try:
        with open(path, "rb") as f:
            f.seek(offset); data = f.read()
    except OSError:
        return [], offset
    cut = data.rfind(b"\n") + 1
    return data[:cut].decode(errors="replace").splitlines(), offset + cut


# ================================================================= provenance

def image_info(image):
    ok, iid = sh(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    if not ok:
        return None, None, f"docker image {image}: {iid.splitlines()[0] if iid else 'not found'}"
    ok, out = sh(["docker", "run", "--rm", image, "python3", "-m", "pip", "freeze"])
    return iid, (out.splitlines() if ok else f"FAILED: {out}"), None


def resolve_commit(repo, ref):
    ok, out = sh(["git", "rev-parse", "--verify", ref + "^{commit}"], repo)
    return out if ok else None


# ================================================================= training (models with a train block)

EP_RE = re.compile(r"^(w\d+) Ep:\s*(\d+)\s*\|\s*Loss:\s*(\S+)\s+(\S+)\s*\|\s*Util:\s*(\S+)\s*\|\s*"
                   r"Reward:\s*(\S+)(?:\s*\|\s*Ent:\s*(\S+))?")
LOSS_ORDER_COMMIT = "eb27ae7"   # from here on traina3c prints (critic, actor); before it, (actor, critic)


def loss_order(repo, commit):
    if not resolve_commit(repo, LOSS_ORDER_COMMIT):
        return "unknown"
    ok, _ = sh(["git", "merge-base", "--is-ancestor", LOSS_ORDER_COMMIT, commit], repo)
    return "critic,actor" if ok else "actor,critic"


def parse_ep(line, order):
    m = EP_RE.match(line.strip())
    if not m:
        return None
    w, ep, x, y, util, rew, ent = m.groups()
    f = lambda v: float(v) if v is not None else None
    actor, critic = (f(y), f(x)) if order == "critic,actor" else (f(x), f(y))
    return {"worker": w, "ep": int(ep), "actor": actor, "critic": critic,
            "util": f(util), "reward": f(rew), "ent": f(ent)}


class Watchdog:
    """Divergence / NaN stop at once; collapse (low entropy, or exact-zero actor loss on old code) warns or stops."""
    def __init__(self, cfg, order):
        self.cfg, self.order, self.recent, self.warned = cfg, order, [], False

    def feed(self, r):
        c = self.cfg
        vals = [r["actor"], r["critic"]] + ([r["ent"]] if r["ent"] is not None else [])
        if any(v != v for v in vals):
            return "stop", f"NaN at ep {r['ep']} ({r['worker']})"
        if self.order != "unknown" and abs(r["actor"]) > c["actor_limit"]:
            return "stop", f"divergence: actor loss {r['actor']:.3g} at ep {r['ep']}, limit {c['actor_limit']:.0e}"
        self.recent = (self.recent + [r])[-c["window"]:]
        if len(self.recent) < c["window"]:
            return None
        ents = [x["ent"] for x in self.recent if x["ent"] is not None]
        if ents:
            collapsed, why = max(ents) < c["collapse_ent"], f"max Ent {max(ents):.2g} over last {c['window']} lines"
        elif self.order != "unknown":
            collapsed, why = all(x["actor"] == 0.0 for x in self.recent), f"actor loss 0.0 for {c['window']} lines"
        else:
            return None
        if collapsed and not self.warned:
            self.warned = c["on_collapse"] == "warn"
            return c["on_collapse"], f"collapse at ep {r['ep']}: {why}"
        if not collapsed:
            self.warned = False
        return None


def check_code_supports(code_dir, ref, ent):
    src = _read(os.path.join(code_dir, "traina3c.py"))
    errs = []
    if '"--slo"' not in src:
        errs.append(f"code {ref} has no --slo/--freq/--budget/--overrun (needs fix5 or later)")
    if ent > 0 and '"--ent"' not in src:
        errs.append(f"code {ref} has no --ent (needs fix6 or later) but train.ent = {ent}")
    npy = os.path.join(code_dir, "load-gen", "arrival_rates", "2")
    if not (os.path.isdir(npy) and any(f.endswith(".npy") for f in os.listdir(npy))):
        errs.append(f"code {ref} has no load-gen/arrival_rates/2/*.npy")
    if errs:
        raise SpecError("code/spec mismatch:\n  - " + "\n  - ".join(errs))
    return '"--ent"' in src


def train_dir(repo, s, m):
    return os.path.join(repo, "runs", f"{s['name']}-{m['label']}")


def train_state(repo, s, m):
    """'done' (reusable), 'failed' (a real result: excluded), or 'todo' (never ran, or was interrupted)."""
    d = train_dir(repo, s, m)
    tj = os.path.join(d, "train.json")
    if not os.path.exists(tj):
        return ("todo", "interrupted before it finished: retraining") if os.path.exists(d) else ("todo", "")
    rec = _load(tj)
    if rec.get("exit_code") == 0 and rec.get("checkpoints"):
        return "done", f"{len(rec['checkpoints'])} checkpoints"
    if str(rec.get("stopped_by", "")).startswith("user"):
        return "todo", "stopped with Ctrl-C last time: retraining"
    return "failed", rec.get("stopped_by") or f"exit code {rec.get('exit_code')}"


def remove_run_dir(repo, d):
    code = os.path.join(d, "code")
    if os.path.exists(code):
        sh(["git", "worktree", "remove", "--force", code], repo)
    shutil.rmtree(d, ignore_errors=True)
    sh(["git", "worktree", "prune"], repo)


def run_training(repo, s, m, poll, image_id, pip):
    t, c = m["train"], s["conditions"]
    run = f"{s['name']}-{m['label']}"
    d = train_dir(repo, s, m)
    commit = resolve_commit(repo, t["code"])
    code, out = os.path.join(d, "code"), os.path.join(d, "out")
    if os.path.exists(d):                     # interrupted earlier (train_state said 'todo'): start clean
        remove_run_dir(repo, d)
    os.makedirs(d)
    make_worktree(repo, commit, code)
    has_ent = check_code_supports(code, t["code"], t["ent"])
    os.makedirs(out); os.makedirs(os.path.join(code, "a3c"), exist_ok=True)
    args = ["-n", run, "-k", str(c["knob"]), "-e", str(t["episodes"]), "-ncf", str(c["flavors"]),
            "-ncp", str(c["chain"]), "-nw", str(s["workers"]), "--slo", str(c["slo"]),
            "--freq", str(freq_of(c)), "--budget", str(c["budget"][0]), str(c["budget"][1]),
            "--overrun", str(c["overrun"])] + (["--ent", str(t["ent"])] if has_ent else [])
    script = load_server_script(s["workers"], f"{C_CODE}/a3c/loadservers.log") + \
        "exec python3 -u traina3c.py " + " ".join(shlex.quote(a) for a in args) + "\n"
    cname = f"counsel-train-{run}"
    cmd = docker_cmd(cname, s["image"], [(code, C_CODE, "ro"), (out, f"{C_CODE}/a3c", "rw")], script,
                     f"training {run}")
    _dump({"code_ref": t["code"], "code_commit": commit, "image_id": image_id, "pip_freeze": pip,
           "train": t, "conditions": c, "cmd": cmd}, os.path.join(d, "provenance.json"))

    order = loss_order(repo, commit)
    wd = Watchdog(s["watch"], order) if s["watch"]["enabled"] else None
    log_path = os.path.join(d, "train.log")
    print(f"  training {run}: code {t['code']} = {commit[:7]}, {t['episodes']} eps, ent {t['ent']}, "
          f"loss columns {order}; log {log_path}")
    t0, rec, events = time.time(), {"start": datetime.now().isoformat(timespec="seconds"), "loss_order": order}, []
    offset, last_new, last = 0, time.time(), None

    def stop(reason):
        print(f"  stopping {cname}: {reason}", flush=True)
        sh(["docker", "stop", cname]); proc.wait(); rec["stopped_by"] = reason

    clear_stale_container(cname)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                time.sleep(poll)
                lines, offset = read_new(log_path, offset)
                verdict = None
                for line in lines:
                    r = parse_ep(line, order)
                    if not r:
                        continue
                    last, last_new = r, time.time()
                    if wd and (v := wd.feed(r)):
                        events.append({"t": round(time.time() - t0, 1), "action": v[0], "reason": v[1]})
                        print(f"  WATCHDOG {v[0].upper()}: {v[1]}", flush=True)
                        if v[0] == "stop":
                            verdict = v[1]; break
                if verdict:
                    stop("watchdog: " + verdict); break
                if wd and time.time() - last_new > 60 * s["watch"]["stall_min"]:
                    stop(f"watchdog: no Ep line for {s['watch']['stall_min']} min"); break
                if last:
                    ent = f" ent {last['ent']:.3g}" if last["ent"] is not None else ""
                    loss = (f"losses {last['actor']:.3g} {last['critic']:.3g}" if order == "unknown"
                            else f"actor {last['actor']:.3g} critic {last['critic']:.3g}")
                    print(f"  [{fmt_dt(time.time() - t0)}] ep {last['ep']:>5}/{t['episodes']}  {loss}"
                          f"  reward {last['reward']:.3f}{ent}", flush=True)
        except KeyboardInterrupt:
            stop("user (Ctrl-C)")
            raise
        finally:
            mdir = os.path.join(out, run, "model")
            rec |= {"end": datetime.now().isoformat(timespec="seconds"), "wall_s": round(time.time() - t0, 1),
                    "exit_code": proc.returncode, "watchdog_events": events,
                    "checkpoints": sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []}
            _dump(rec, os.path.join(d, "train.json"))
    return rec


# ================================================================= models -> checkpoints

def model_dir(repo, s, m):
    if m["train"]:
        run = f"{s['name']}-{m['label']}"
        return os.path.join(train_dir(repo, s, m), "out", run, "model"), os.path.join(train_dir(repo, s, m), "out", run)
    base = os.path.join(repo, m["from"])
    return (os.path.join(base, "model"), base) if os.path.isdir(os.path.join(base, "model")) else (base, base)


def checkpoints_of(repo, s, m):
    mdir, _ = model_dir(repo, s, m)
    have = {}
    if os.path.isdir(mdir):
        for f in os.listdir(mdir):
            if (x := re.fullmatch(r"model(\d+)\.pt", f)):
                have[int(x.group(1))] = f
    want = sorted(have) if m["checkpoints"] == "all" else m["checkpoints"]
    return mdir, have, want


def train_conditions(repo, s, m):
    _, run_root = model_dir(repo, s, m)
    p = os.path.join(run_root, "train_conditions.json")
    return _load(p) if os.path.exists(p) else None


def condition_mismatch(tc, c):
    if tc is None:
        return "no train_conditions.json: training conditions unknown (pre-fix5 run?)"
    diffs = []
    for k, v in (("slo", c["slo"]), ("freq", freq_of(c)), ("budget", c["budget"]), ("overrun", c["overrun"]),
                 ("knob", c["knob"]), ("nconf", c["flavors"]), ("ncomp", c["chain"])):
        if k in tc and tc[k] != v:
            diffs.append(f"{k} trained {tc[k]} vs eval {v}")
    return "; ".join(diffs) or None


# ================================================================= eval (inside the container)

EVAL_DRIVER = r'''
"""Written by counselctl.py. Runs eval_harness.run_episode for every (arm, seed) with conditions from the
spec, one process per load-server port, resumable. Uses the eval code mounted at /counsel unchanged."""
import inspect, json, os, sys, time, multiprocessing as mp
with open(sys.argv[1]) as _f:
    cfg = json.load(_f)
sys.path[:0] = [cfg["code"], cfg["code"] + "/tools"]
import eval_harness as eh


def worker(port, jobs, results, cfg):
    from eval_arms import make_arm
    arms = {}
    while (job := jobs.get()) is not None:
        name, spec, seed = job
        try:
            if name not in arms:
                arms[name] = make_arm(spec, **({"vllm_base": cfg["vllm_base"]} if cfg.get("vllm_base") else {}))
            t0 = time.time()
            row = eh.run_episode(arms[name], seed, port, cfg["nconf"], cfg["ncomp"], cfg["R"],
                                 os.path.join(cfg["envlogs"], f"p{port}"))
            row.update(arm=name, spec=spec, seed=seed, port=port, wall_s=time.time() - t0)
        except Exception as e:
            import traceback
            row = dict(arm=name, spec=spec, seed=seed, port=port, error=repr(e), tb=traceback.format_exc()[-1500:])
        results.put(row)
    results.put(None)


def main():
    sig = list(inspect.signature(eh.run_episode).parameters)[:7]
    if sig != ["arm", "seed", "port", "nconf", "ncomp", "R", "log_dir"]:
        sys.exit(f"COUNSELCTL-EVAL: eval_harness.run_episode signature changed: {sig}")
    path = cfg["episodes"]
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [json.loads(l) for l in f if l.strip()]
    have = {(r["arm"], r["seed"]) for r in old if "error" not in r}
    todo = [(n, s, sd) for sd in cfg["seeds"] for n, s in cfg["arms"] if (n, sd) not in have]
    total = len(todo)
    print(f"COUNSELCTL-EVAL: {total} episodes to run, {len(have)} already done", flush=True)
    if not todo:
        return
    if cfg.get("vllm_base") and any(n.startswith("slm_") for n, _, _ in todo):
        import urllib.request
        try:
            with urllib.request.urlopen(cfg["vllm_base"] + "/v1/models", timeout=10) as r:
                ids = [m["id"] for m in json.loads(r.read())["data"]]
        except Exception as e:
            sys.exit(f"COUNSELCTL-EVAL: vLLM not reachable from the container at {cfg['vllm_base']}: {e!r}. "
                     "Is vllm serve listening on all interfaces (its default) and not blocked by a firewall?")
        print(f"COUNSELCTL-EVAL: vLLM reachable at {cfg['vllm_base']}, serving {ids}", flush=True)
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    for j in todo: jobs.put(j)
    for _ in cfg["ports"]: jobs.put(None)
    procs = [ctx.Process(target=worker, args=(p, jobs, results, cfg)) for p in cfg["ports"]]
    for p in procs: p.start()
    alive, done = len(procs), 0
    with open(path, "a") as f:
        while alive:
            r = results.get()
            if r is None:
                alive -= 1; continue
            f.write(json.dumps(r) + "\n"); f.flush()
            done += 1
            tag = "ERROR " + r["error"] if "error" in r else f"breaches {r['breaches']}"
            print(f"COUNSELCTL-EVAL: {done}/{total} {r['arm']} seed {r['seed']} {tag}", flush=True)
    for p in procs: p.join()


if __name__ == "__main__":
    main()
'''


def arms_for(repo, s, plan):
    """[(name, spec, host_ckpt_path or None)], baselines first."""
    arms = [(b.replace(":", "_").replace(".", ""), b, None) for b in s["eval"]["baselines"]]
    for m, mdir, want in plan:
        for ep in want:
            for mode in s["eval"]["modes"]:
                arms.append((f"{m['label']}_m{ep}_{mode[0]}", f"torch:{C_CKPT}/{m['label']}/model{ep}.pt:{mode}",
                             os.path.join(mdir, f"model{ep}.pt")))
    for x in s.get("slm", []):
        arms.append((f"slm_{x['label']}", f"slm:{x['model']}:{x['prompt']}", None))
    return arms


def run_eval(repo, s, job, out, arms, mounts_ckpt, poll):
    c = s["conditions"]
    seeds = parse_seeds(s["eval"]["seeds"])
    ports = [8000 + i for i in range(s["workers"])]
    cfg = {"arms": [[n, sp] for n, sp, _ in arms], "seeds": seeds, "ports": ports,
           "nconf": c["flavors"], "ncomp": c["chain"], "envlogs": f"{C_JOB}/envlogs", "code": C_CODE,
           "episodes": f"{C_RESULTS}/episodes.jsonl",
           "R": {"slo": c["slo"], "freq": freq_of(c), "knob": c["knob"], "budget": c["budget"], "slack": c["overrun"]},
           "vllm_base": container_base(s["vllm"]["base"]) if s.get("slm") else None}
    _dump(cfg, os.path.join(job, "eval_config.json"))
    _write(os.path.join(job, "eval_driver.py"), EVAL_DRIVER)
    script = load_server_script(s["workers"], f"{C_JOB}/loadservers.log") + \
        f"exec python3 -u {C_JOB}/eval_driver.py {C_JOB}/eval_config.json\n"
    mounts = [(os.path.join(job, "code"), C_CODE, "ro"), (job, C_JOB, "rw"), (out, C_RESULTS, "rw")] + mounts_ckpt
    cname = f"counsel-eval-{s['name']}"
    extra = ["--add-host", "host.docker.internal:host-gateway"] if s.get("slm") else []
    cmd = docker_cmd(cname, s["image"], mounts, script, f"evaluating {s['name']}", extra)
    log_path = os.path.join(job, "eval.log")
    todo = len(seeds) * len(arms)
    print(f"  evaluating {len(arms)} arms x {len(seeds)} seeds = {todo} episodes on {len(ports)} workers; "
          f"log {log_path}")
    clear_stale_container(cname)
    stall = 60 * s["watch"]["stall_min"]
    t0, last, shown, t_up, last_progress, reason = time.time(), "", "", None, time.time(), None
    offset = os.path.getsize(log_path) if os.path.exists(log_path) else 0     # resumed runs append to the log
    log_start = offset
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                time.sleep(poll)
                lines, offset = read_new(log_path, offset)
                for line in lines:
                    if not line.startswith("COUNSELCTL"):
                        continue
                    last = line.split(": ", 1)[-1]
                    if "episodes to run" in line:               # always show what a (re)run will do
                        print(f"  {last}", flush=True)
                    if "load servers up" in line:
                        t_up = time.time()
                    if (x := re.match(r"COUNSELCTL-EVAL: (\d+)/(\d+) (.*)", line)):
                        k, n, what = int(x.group(1)), int(x.group(2)), x.group(3)
                        last_progress = time.time()
                        rate = k / max(1e-6, time.time() - (t_up or t0))   # episodes/s since the servers came up
                        last = f"{k}/{n} episodes, ETA {fmt_dt((n - k) / rate)}  (last: {what})"
                if last and last != shown:
                    print(f"  [{fmt_dt(time.time() - t0)}] {last}", flush=True); shown = last
                if time.time() - last_progress > stall:
                    reason = f"no episode finished for {s['watch']['stall_min']} min (stalled)"
                    print(f"  stopping {cname}: {reason}; re-run the same command to resume", flush=True)
                    sh(["docker", "stop", cname]); proc.wait()
                    break
        except KeyboardInterrupt:
            print(f"\n  stopping {cname} (re-run the same command to resume)")
            sh(["docker", "stop", cname]); proc.wait()
            raise
    rc = proc.returncode
    if rc != 0 and not reason:
        tail = log_tail(log_path, log_start, 15)
        reason = f"eval container exited with code {rc}"
        print(f"  {reason}; last lines of {os.path.basename(log_path)}:\n" + "\n".join("    " + l for l in tail))
    return rc, reason


def log_tail(path, start, n):
    try:
        with open(path, "rb") as f:
            f.seek(start); lines = f.read().decode(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def clear_stale_container(cname):
    """A container left over from a crash (same name) would make `docker run --name` fail."""
    ok, out = sh(["docker", "ps", "-aq", "--filter", f"name=^{cname}$"])
    if ok and out.strip():
        print(f"  removing leftover container {cname} from an earlier run")
        sh(["docker", "rm", "-f", cname])


# ================================================================= report (host, stdlib)

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00, 120: 1.98}


def t975(df):
    return T975[max(k for k in T975 if k <= df)] if df >= 1 else float("nan")


def ci(xs):
    if len(xs) < 2:
        return float("nan")
    return t975(len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs))


def load_rows(path):
    rows = [json.loads(l) for l in _read(path).splitlines() if l.strip()] if os.path.exists(path) else []
    ok, seen = [], set()
    for r in rows:                     # first successful row per (arm, seed); duplicates can't double-count
        if "error" not in r and (r["arm"], r["seed"]) not in seen:
            ok.append(r); seen.add((r["arm"], r["seed"]))
    bad, bseen = [], set()
    for r in rows:
        k = (r["arm"], r["seed"])
        if "error" in r and k not in seen and k not in bseen:
            bad.append(r); bseen.add(k)
    done = seen
    bad = [r for r in bad if (r["arm"], r["seed"]) not in done]      # a failure later retried successfully
    return ok, bad


def svg_plot(s, plan, by, path):
    """Mean breaches per checkpoint: one line per model x mode, dashed lines for baselines. Plain SVG."""
    series = []
    for m, _, want in plan:
        for mode in s["eval"]["modes"]:
            pts = [(ep, st.mean(by[a]["breaches"].values())) for ep in want
                   if (a := f"{m['label']}_m{ep}_{mode[0]}") in by]
            if pts:
                series.append((f"{m['label']} {mode}", pts, mode == "greedy"))
    base = [(n, st.mean(v["breaches"].values())) for n, v in by.items() if v["baseline"]]
    if not series:
        return False
    xs = [x for _, p, _ in series for x, _ in p]
    ys = [y for _, p, _ in series for _, y in p] + [y for _, y in base]
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y1 = max(ys) * 1.08 or 1
    W, H, L, R, T, B = 760, 420, 60, 170, 20, 45
    px = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
    py = lambda y: T + (1 - y / y1) * (H - T - B)
    pal = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f"]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="13" font-size="12">{s["name"]}: mean SLO breaches per episode ({len(parse_seeds(s["eval"]["seeds"]))} seeds)</text>']
    for k in range(6):
        y = y1 * k / 5
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="#eee"/>'
                 f'<text x="{L - 6}" y="{py(y) + 4:.1f}" text-anchor="end">{y:.0f}</text>')
    for x in sorted(set(xs)):
        o.append(f'<text x="{px(x):.1f}" y="{H - B + 15}" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">checkpoint (training episode)</text>')
    legend_y = T + 10
    for i, (n, y) in enumerate(sorted(base, key=lambda b: b[1])):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="#888" stroke-dasharray="4 3"/>'
                 f'<text x="{W - R + 4}" y="{py(y) + 4:.1f}" fill="#666">{n} {y:.0f}</text>')
    for i, (n, pts, greedy) in enumerate(series):
        col = pal[i // 2 % len(pal)] if len(s["eval"]["modes"]) == 2 else pal[i % len(pal)]
        dash = "" if greedy else ' stroke-dasharray="1 0"'
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        o.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="{2 if greedy else 1.2}"'
                 f' opacity="{1 if greedy else 0.65}"{dash}/>')
        o += [f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{col}"/>' for x, y in pts]
        o.append(f'<text x="{L + 8}" y="{legend_y + 13 * i}" fill="{col}">{"━" if greedy else "─"} {n}</text>')
    o.append("</svg>")
    _write(path, "\n".join(o))
    return True


def build_report(repo, s, out, plan, prov, train_notes, warnings):
    rows, bad = load_rows(os.path.join(out, "episodes.jsonl"))
    seeds = parse_seeds(s["eval"]["seeds"])
    base_names = {b.replace(":", "_").replace(".", ""): b for b in s["eval"]["baselines"]}
    by = {}
    for r in rows:
        if r["seed"] not in seeds:
            continue
        a = by.setdefault(r["arm"], {"baseline": r["arm"] in base_names, "breaches": {}, "rows": {}})
        a["breaches"][r["seed"]] = r["breaches"]; a["rows"][r["seed"]] = r

    def mean(arm, k):
        v = [x[k] for x in by[arm]["rows"].values() if x.get(k) is not None]
        return st.mean(v) if v else float("nan")

    # ties broken by name: worker completion order must not change the report between identical runs
    order = sorted(by, key=lambda a: (st.mean(by[a]["breaches"].values()), a))
    fail_by = {}
    for r in bad:
        if r["seed"] in seeds:
            fail_by[r["arm"]] = fail_by.get(r["arm"], 0) + 1
    # behaviour fingerprint per seed: identical on every seed = the same policy in practice
    fp = {a: {sd: (r["breaches"], r["applied"], r["cpu_mean"], r["mem_mean"]) for sd, r in by[a]["rows"].items()}
          for a in by}
    same_as = {}
    for a in order:
        if by[a]["baseline"]:
            continue
        match = [b for b in base_names if b in fp and fp[a] and fp[a] == fp[b]]
        if match:
            same_as[a] = " / ".join(match)

    L = [f"# {s['name']}", "",
         f"Generated {datetime.now().isoformat(timespec='seconds')} by `tools/counselctl.py run`. "
         f"All numbers below are measured, from `episodes.jsonl` in this folder.", ""]
    c = s["conditions"]
    L += ["## Setup", "",
          f"- Conditions: SLO {c['slo']}, freq {freq_of(c)}, budget {c['budget']}, overrun {c['overrun']}, "
          f"knob {c['knob']}, flavors {c['flavors']}, chain {c['chain']}",
          f"- Seeds: {s['eval']['seeds']} ({len(seeds)}); modes: {', '.join(s['eval']['modes'])}; "
          f"workers: {s['workers']}",
          f"- Eval code: `{prov['eval_commit'][:7]}` ({prov['tool_describe']}); image `{(prov['image_id'] or '?')[:19]}`",
          f"- Episodes: {sum(len(v['breaches']) for v in by.values())} done, {len(bad)} failed", ""]
    L += ["## Models", "", "| label | source | checkpoints | trained on | note |", "|---|---|---|---|---|"]
    in_plan = {m["label"]: want for m, _, want in plan}
    for m in s["models"]:                                   # every model, including excluded ones
        tc = train_conditions(repo, s, m) if m["label"] in in_plan else None
        tc_s = f"SLO {tc.get('slo')}, freq {tc.get('freq')}, budget {tc.get('budget')}" if tc else "unknown"
        src = f"trained here ({m['train']['code']}, ent {m['train']['ent']})" if m["train"] else f"`{m['from']}`"
        ck = ", ".join(map(str, in_plan[m["label"]])) if m["label"] in in_plan else "not evaluated"
        L.append(f"| {m['label']} | {src} | {ck} | {tc_s} | {train_notes.get(m['label'], '')} |")
    L.append("")
    if warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    L += ["## Results (sorted by mean breaches; lower is better)", "",
          "| arm | n | failed | breaches | 95% CI | reward/step | applied | util | overrun steps | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        b = list(by[a]["breaches"].values())
        h = ci(b)
        note = "baseline" if by[a]["baseline"] else (f"= {same_as[a]} on every seed" if a in same_as else "")
        L.append(f"| {a} | {len(b)} | {fail_by.get(a, 0)} | {st.mean(b):.1f} | ±{h:.1f} | "
                 f"{mean(a, 'reward_per_step'):.3f} | {mean(a, 'applied'):.1f} | {mean(a, 'util_mean'):.3f} | "
                 f"{mean(a, 'overrun_steps'):.1f} | {note} |")
    L.append("")
    dead = sorted(set(fail_by) - set(by))
    if dead:                          # every episode of these arms failed: they are not in the table above
        L += ["## Arms with no successful episode", "", "| arm | failed | first error |", "|---|---|---|"]
        for a in dead:
            err = next(r["error"] for r in bad if r["arm"] == a)
            L.append(f"| {a} | {fail_by[a]} | `{err[:160]}` |")
        L.append("")
    L += ["CI = 95% t-interval of the mean over seeds. `= X on every seed` means breaches, actions applied, "
          "and mean CPU/memory match baseline X exactly on every seed: in practice the same policy.", ""]

    slm_arms = [a for a in order if a.startswith("slm_")]
    slm_bad = []
    if slm_arms:
        L += ["## SLM arms (served by vLLM)", "",
              "| arm | served model | queries/episode | invalid | errors | decision ms (mean / p95) |",
              "|---|---|---|---|---|---|"]
        for a in slm_arms:
            rs = list(by[a]["rows"].values())
            q = sum(r.get("llm_queries") or 0 for r in rs)
            inv = sum(r.get("llm_invalid") or 0 for r in rs)
            er = sum(r.get("llm_errors") or 0 for r in rs)
            model = str(rs[0].get("spec", "")).split(":", 1)[-1]
            if q:
                L.append(f"| {a} | `{model}` | {q / len(rs):.1f} | {inv / q:.1%} | {er / q:.1%} | "
                         f"{mean(a, 'llm_lat_ms_mean'):.0f} / {mean(a, 'llm_lat_ms_p95'):.0f} |")
                if er / q > 0.01:
                    slm_bad.append(f"{a}: {er / q:.1%} of LLM requests failed and fell back to no change")
            else:
                L.append(f"| {a} | `{model}` | 0 | - | - | - |")
        L += ["", "Invalid = the reply was not a listed option number; errors = the request failed. Both fall back "
              "to action 0 (no change). vLLM at temperature 0 is not guaranteed bit-exact across runs or batch "
              "sizes, so SLM rows can differ slightly on a re-run.", ""]

    refs = [r.replace(":", "_").replace(".", "") for r in s["eval"]["reference"]]
    refs = [r for r in refs if r in by]
    if refs:
        L += ["## Paired comparison vs reference arms", "",
              "Same seeds, so each difference is on the same load trace. diff = arm minus reference "
              "(negative = fewer breaches). `*` = 95% interval excludes 0. W/T/L = seeds where the arm had "
              "fewer / equal / more breaches. No correction for the number of comparisons.", "",
              "| arm | " + " | ".join(f"vs {r}" for r in refs) + " |", "|---|" + "---|" * len(refs)]
        for a in order:
            if a in refs:
                continue
            cells = []
            for r in refs:
                sd = sorted(set(by[a]["breaches"]) & set(by[r]["breaches"]))
                d = [by[a]["breaches"][x] - by[r]["breaches"][x] for x in sd]
                if len(d) < 2:
                    cells.append("-"); continue
                m, h = st.mean(d), ci(d)
                w = sum(x < 0 for x in d); t = sum(x == 0 for x in d); l = sum(x > 0 for x in d)
                cells.append(f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]{' *' if (m - h > 0 or m + h < 0) else ''} {w}/{t}/{l}")
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")

    over = [a for a in by if mean(a, "overrun_steps") > 0]
    L += ["## Automatic notes", ""]
    if same_as:
        L.append(f"- {len(same_as)} of {sum(not v['baseline'] for v in by.values())} checkpoint arms behave exactly "
                 f"like a baseline on every seed: " + ", ".join(f"{a} = {b}" for a, b in list(same_as.items())[:12])
                 + (" ..." if len(same_as) > 12 else ""))
    best = next((a for a in order if not by[a]["baseline"]), None)
    if best:
        L.append(f"- Best checkpoint arm: {best}, {st.mean(by[best]['breaches'].values()):.1f} breaches "
                 f"(best baseline: {next((a for a in order if by[a]['baseline']), '-')})")
    if not over:
        L.append(f"- No arm ever exceeded the budget (overrun steps 0 everywhere): budget {c['budget']} does not bind "
                 f"under these conditions.")
    for x in slm_bad:
        L.append(f"- WARNING {x}: those episodes do not measure the model")
    if bad:
        L.append(f"- {len(bad)} episode(s) failed; first error: `{bad[0]['error'][:200]}`")
    L.append("")
    if svg_plot(s, plan, by, os.path.join(out, "breaches.svg")):
        L += ["## Breaches per checkpoint", "", "![breaches per checkpoint](breaches.svg)", ""]

    L += ["## Provenance", "", "```", json.dumps({k: v for k, v in prov.items() if k != "pip_freeze"}, indent=1),
          "```", "Full `pip freeze` of the image is in `provenance.json`.", ""]
    _write(os.path.join(out, "report.md"), "\n".join(L))

    csv = ["arm,n,breaches_mean,breaches_ci95,reward_per_step,applied,util_mean,overrun_steps,same_as"]
    for a in order:
        b = list(by[a]["breaches"].values())
        csv.append(f"{a},{len(b)},{st.mean(b):.4g},{ci(b):.4g},{mean(a, 'reward_per_step'):.4g},"
                   f"{mean(a, 'applied'):.4g},{mean(a, 'util_mean'):.4g},{mean(a, 'overrun_steps'):.4g},{same_as.get(a, '')}")
    _write(os.path.join(out, "summary.csv"), "\n".join(csv) + "\n")
    return order, by, bad


def append_runs_md(repo, s, out, prov, order, by, bad):
    p = os.path.join(repo, "results", "RUNS.md")
    if not os.path.exists(p):
        return "results/RUNS.md not found: not updated"
    txt = _read(p)
    head = "## Evaluations: counselctl"
    rel = os.path.relpath(out, repo)
    best = next((a for a in order if not by[a]["baseline"]), "-")
    bm = lambda a: f"{st.mean(by[a]['breaches'].values()):.1f}" if a in by else "-"
    row = (f"| {s['name']} | {prov['eval_commit'][:7]} | {(prov['image_id'] or '?')[7:19]} | "
           f"SLO {s['conditions']['slo']}, budget {s['conditions']['budget']}, seeds {s['eval']['seeds']} | "
           f"{', '.join(m['label'] for m in s['models'])} | {rel}/report.md | "
           f"[measured] best ckpt {best} {bm(best)}; random {bm('random')}; heur {bm('heur')}; {len(bad)} failed |")
    lines = txt.rstrip("\n").split("\n")
    if head not in txt:
        lines += ["", head, "", "Rows written by `tools/counselctl.py run`; the report has the full table.", "",
                  "| Eval | Eval code | Image | Conditions | Models | Report | Headline |", "|---|---|---|---|---|---|---|"]
    tag = f"| {s['name']} |"
    idx = [i for i, l in enumerate(lines) if l.startswith(tag)]
    if idx:
        lines[idx[0]] = row
    else:
        # insert after the last row of the counselctl table
        start = lines.index(head)
        end = start + 1
        while end < len(lines) and not lines[end].startswith("## "):
            end += 1
        k = end
        while k > start and not lines[k - 1].startswith("|"):
            k -= 1
        lines.insert(k, row)
    _write(p, "\n".join(lines) + "\n")
    return f"results/RUNS.md: row for {s['name']} written (commit it)"


# ================================================================= run

def cmd_run(a):
    repo = os.path.abspath(a.repo)
    spec_path = os.path.abspath(a.spec)
    s = load_spec(spec_path)
    out = os.path.join(repo, "results", "eval", s["name"])
    job = os.path.join(repo, "runs", s["name"])
    problems, warnings = [], []

    # --- 1. resume rules: same name = same experiment (only eval.seeds may change)
    resolved_p = os.path.join(out, "spec.resolved.json")
    if os.path.exists(out) and not os.path.exists(resolved_p):
        raise SpecError(f"{out} exists but was not made by counselctl: pick another name")
    if os.path.exists(resolved_p):
        old = _load(resolved_p)
        strip = lambda x: {**x, "eval": {**x["eval"], "seeds": None},
                           "vllm": x.get("vllm", TOP["vllm"]), "slm": x.get("slm", [])}
        if strip(old) != strip(s):
            raise SpecError(f"{s['name']} already exists with different settings (only eval.seeds may change "
                            f"on a re-run): use a new name")
        print(f"resuming {s['name']}")

    # --- 2. code, image, models
    eval_commit = resolve_commit(repo, "HEAD")
    # describes the commit the eval code comes from (a worktree of it), so no "-dirty": uncommitted eval-code
    # edits get their own warning below, and an edited RUNS.md does not change what ran
    _, tool_describe = sh(["git", "describe", "--tags", "--always", eval_commit or "HEAD"], repo)
    if not eval_commit:
        problems.append(f"{repo} is not a git repository")
    # counselctl.py itself runs on the host and is not eval code; its own hash is in provenance.json
    _, dirty = sh(["git", "status", "--porcelain", "--untracked-files=no", "--", "tools", "model", "load-gen",
                   ":(exclude)tools/counselctl.py"], repo)
    if dirty:
        warnings.append(f"uncommitted changes in tools/model/load-gen are NOT used (eval code comes from commit "
                        f"{(eval_commit or '?')[:7]}): {dirty.splitlines()}")
    iid, pip, err = image_info(s["image"])
    if err:
        problems.append(err)
    # const:CxM baselines must name a real shape, or every episode of that arm fails at run time
    nf = s["conditions"]["flavors"]
    ok, fj = sh(["git", "show", f"HEAD:model/configs/flavors{nf}.json"], repo)
    if ok:
        shapes = {tuple(v) for k, v in json.loads(fj).items() if k != "none"}
        for b in s["eval"]["baselines"]:
            if (x := re.fullmatch(r"const:(\d+)x(\d+)", b)) and tuple(map(int, x.groups())) not in shapes:
                problems.append(f"baseline {b}: no such shape in flavors{nf}.json; have "
                                + ", ".join(f"const:{c}x{m}" for c, m in sorted(shapes)))
    else:
        problems.append(f"model/configs/flavors{nf}.json not found at HEAD")
    if s["slm"]:
        try:
            served = [m["id"] for m in vllm_models(s["vllm"]["base"])]
            for x in s["slm"]:
                if x["model"] not in served:
                    problems.append(f"slm {x['label']}: '{x['model']}' is not served at {s['vllm']['base']} "
                                    f"(serving {served}); one vllm serve = one model, so put one [[slm]] per run")
        except Exception as ex:
            problems.append(f"vLLM not reachable at {s['vllm']['base']} ({ex!r}): start vllm serve first")
        ok, arms_src = sh(["git", "show", "HEAD:tools/eval_arms.py"], repo)
        for x in s["slm"]:
            if ok and f'"{x["prompt"]}":' not in arms_src:
                problems.append(f"slm {x['label']}: prompt '{x['prompt']}' is not a key of PROMPTS in eval_arms.py")

    plan, train_notes, to_train = [], {}, []
    for m in s["models"]:
        if m["train"]:
            if not resolve_commit(repo, m["train"]["code"]):
                problems.append(f"{m['label']}: train.code '{m['train']['code']}' is not a commit/tag/branch")
                continue
            state, why = train_state(repo, s, m)
            if state == "done":
                train_notes[m["label"]] = f"training reused ({why})"
            elif state == "failed":
                train_notes[m["label"]] = f"training failed earlier: {why}; excluded"
                warnings.append(f"{m['label']}: earlier training failed ({why}). Delete "
                                f"{os.path.relpath(train_dir(repo, s, m), repo)} to retry.")
                continue
            elif a.no_train:
                train_notes[m["label"]] = "not trained yet (--no-train); excluded"
                continue
            else:
                to_train.append(m)
                train_notes[m["label"]] = "trained in this run" + (f" ({why})" if why else "")
            plan.append(m)
            continue
        mdir, have, want = checkpoints_of(repo, s, m)
        if not have:
            problems.append(f"{m['label']}: no modelN.pt under {m['from']}")
            continue
        missing = [e for e in want if e not in have]
        if missing:
            problems.append(f"{m['label']}: checkpoints {missing} not in {os.path.relpath(mdir, repo)} "
                            f"(have {sorted(have)})")
            continue
        plan.append(m)

    # --- 3. plan
    seeds = parse_seeds(s["eval"]["seeds"])
    print(f"{s['name']}: SLO {s['conditions']['slo']}, freq {freq_of(s['conditions'])}, budget "
          f"{s['conditions']['budget']}, {len(seeds)} seeds, {s['workers']} workers")
    print(f"eval code {(eval_commit or '?')[:7]} ({tool_describe}), image {(iid or '?')[:19]}")
    for m in s["models"]:
        src = f"train {m['train']['code']} {m['train']['episodes']} eps ent {m['train']['ent']}" if m["train"] else m["from"]
        print(f"  model {m['label']:<12} {src:<40} {train_notes.get(m['label'], '')}")
    for x in s["slm"]:
        print(f"  slm   {x['label']:<12} {x['model']} (prompt {x['prompt']}) via {s['vllm']['base']}")
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        print("PROBLEMS:\n  - " + "\n  - ".join(problems))
        return 1
    if a.check:
        n_ck = sum(len(checkpoints_of(repo, s, m)[2]) for m in plan if not m["train"])
        print(f"\nOK: would train {len(to_train)} model(s), then evaluate {len(s['eval']['baselines'])} baselines + "
              f"{n_ck} existing checkpoints x {len(s['eval']['modes'])} modes (+ any trained) x {len(seeds)} seeds")
        return 0

    os.makedirs(out, exist_ok=True)
    os.makedirs(job, exist_ok=True)
    lock = os.path.join(job, "running.pid")          # one run per name at a time: both would append to one file
    if os.path.exists(lock):
        pid = int(_read(lock) or 0)
        try:
            os.kill(pid, 0)
            raise SpecError(f"{s['name']} is already running (pid {pid}); wait for it or stop it first")
        except ProcessLookupError:
            pass                                      # stale lock from a crashed run
    _write(lock, str(os.getpid()))
    try:
        return _run_locked(a, repo, spec_path, s, out, job, resolved_p, plan, to_train, train_notes,
                           warnings, iid, pip, eval_commit, tool_describe)
    finally:
        os.remove(lock)


def _run_locked(a, repo, spec_path, s, out, job, resolved_p, plan, to_train, train_notes,
                warnings, iid, pip, eval_commit, tool_describe):
    _dump(s, resolved_p)
    shutil.copy(spec_path, os.path.join(out, "spec.toml"))

    # --- 4. train what needs training
    failed_training = []
    for m in to_train:
        rec = run_training(repo, s, m, a.poll, iid, pip)
        if rec.get("exit_code") != 0 or not rec.get("checkpoints"):
            why = rec.get("stopped_by") or f"exit code {rec.get('exit_code')}"
            if not rec.get("checkpoints") and not rec.get("stopped_by"):
                why += ", no checkpoints written"
            train_notes[m["label"]] = f"training failed: {why}; excluded"
            warnings.append(f"{m['label']}: training failed ({why}); see "
                            f"{os.path.relpath(train_dir(repo, s, m), repo)}/train.log")
            tail = log_tail(os.path.join(train_dir(repo, s, m), "train.log"), 0, 12)
            print(f"  TRAINING FAILED for {m['label']}: {why}; it is excluded from the eval. Last lines of train.log:\n"
                  + "\n".join("    " + l for l in tail), flush=True)
            failed_training.append(m["label"])
            plan.remove(m)
        else:
            train_notes[m["label"]] = f"trained in this run ({fmt_dt(rec['wall_s'])}, {len(rec['checkpoints'])} ckpts)"

    # --- 5. checkpoints: resolve, hash, compare training conditions
    plan3, mounts, ckpts = [], [], {}
    for m in plan:
        mdir, have, want = checkpoints_of(repo, s, m)
        want = [e for e in want if e in have]
        if not want:
            warnings.append(f"{m['label']}: no checkpoints found after training; excluded")
            continue
        if (mm := condition_mismatch(train_conditions(repo, s, m), s["conditions"])):
            warnings.append(f"{m['label']}: eval conditions differ from training: {mm}")
        plan3.append((m, mdir, want))
        mounts.append((mdir, f"{C_CKPT}/{m['label']}", "ro"))
        for e in want:
            ckpts[f"{m['label']}/model{e}.pt"] = sha256(os.path.join(mdir, f"model{e}.pt"))
    arms = arms_for(repo, s, plan3)

    # --- 6. eval code worktree at HEAD (recreated each run; outputs live elsewhere)
    code = os.path.join(job, "code")
    if os.path.exists(code):
        sh(["git", "worktree", "remove", "--force", code], repo)
        shutil.rmtree(code, ignore_errors=True)
    make_worktree(repo, eval_commit, code)
    prov = {"created": datetime.now().isoformat(timespec="seconds"), "host": os.uname().nodename,
            "eval_commit": eval_commit, "tool_describe": tool_describe,
            "counselctl_sha256": sha256(os.path.abspath(__file__)), "image": s["image"], "image_id": iid,
            "checkpoint_sha256": ckpts, "pip_freeze": pip}
    if s["slm"]:
        try:
            prov["vllm"] = {"base": s["vllm"]["base"], "container_base": container_base(s["vllm"]["base"]),
                            "served": vllm_models(s["vllm"]["base"])}
        except Exception as ex:
            prov["vllm"] = {"base": s["vllm"]["base"], "error": repr(ex)}
    old_prov = os.path.join(out, "provenance.json")
    if os.path.exists(old_prov):
        prev = _load(old_prov)
        for k in ("eval_commit", "image_id"):
            if prev.get(k) and prev[k] != prov[k]:
                warnings.append(f"resumed with a different {k} ({prev[k][:12]} before, {prov[k][:12]} now): "
                                f"episodes in this folder come from both")
        changed = [k for k, v in ckpts.items() if prev.get("checkpoint_sha256", {}).get(k, v) != v]
        if changed:
            warnings.append(f"checkpoint files changed since the last run: {changed}")
    _dump(prov, old_prov)

    # --- 7. evaluate, then report
    rc, reason = run_eval(repo, s, job, out, arms, mounts, a.poll)
    if rc != 0:
        warnings.append(f"{reason}; see {os.path.relpath(job, repo)}/eval.log. Re-run the same command to resume.")
    order, by, bad = build_report(repo, s, out, plan3, prov, train_notes, warnings)
    note = append_runs_md(repo, s, out, prov, order, by, bad) if not a.no_runs_md else "RUNS.md not updated"
    print(f"\nreport: {os.path.relpath(os.path.join(out, 'report.md'), repo)}\n{note}")
    best = next((x for x in order if not by[x]["baseline"]), None)
    if best:
        print(f"best checkpoint arm {best}: {st.mean(by[best]['breaches'].values()):.1f} breaches; "
              + ", ".join(f"{b} {st.mean(by[b]['breaches'].values()):.1f}" for b in order if by[b]["baseline"]))
    if failed_training:
        print(f"training failed for: {', '.join(failed_training)} (see the report's Warnings)")
    return 0 if rc == 0 and not bad and not failed_training else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=".", help="counsel-a3c-fresh root (default: current dir)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train (if asked), evaluate, write results/eval/<name>/report.md")
    r.add_argument("spec")
    r.add_argument("--check", action="store_true", help="validate and print the plan; run nothing")
    r.add_argument("--no-train", action="store_true", help="evaluate only what already exists")
    r.add_argument("--no-runs-md", action="store_true", help="don't write a row to results/RUNS.md")
    r.add_argument("--poll", type=float, default=30.0, help="seconds between progress lines")
    r.set_defaults(fn=cmd_run)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except SpecError as ex:
        print(ex, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; re-run the same command to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
