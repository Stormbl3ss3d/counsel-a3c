"""Builds a throwaway git repo that looks like counsel-a3c-fresh to counselctl.py, with stub code instead of
the real environment, so the integration tests run in seconds and need no Docker, torch or load data.

Stub behaviour (all deterministic per (arm, seed)):
  eval_harness.run_episode  checks the load server answers on its port, then returns made-up breaches:
                            noop ~380, random ~100, heur ~45, const/largest ~150, checkpoints ~95-102,
                            a checkpoint file containing "const" behaves exactly like const:1x2.
  traina3c.py               prints fix6-style Ep lines and writes model{2,4,...}.pt + train_conditions.json.
Failure switches (environment variables, passed through by the fake docker):
  STUB_FAIL_SEED=1001       every checkpoint arm fails on that seed
  STUB_BAD_CKPT=1           checkpoint arms whose file says "broken" always fail
  STUB_HANG=1               run_episode sleeps forever (stall test)
  STUB_BADSIG=1             run_episode has the wrong signature (driver refuses to start)
  STUB_DIVERGE=1            training actor loss explodes after episode 3
  STUB_TRAIN_CRASH=1        training crashes at start (exit 1, no checkpoints)
  STUB_TRAIN_SLOW=0.2       seconds per training episode
  STUB_EP_SLOW=0.0          seconds per eval episode
"""
import json, os, subprocess


def _w(p, txt):
    with open(p, "w") as f:
        f.write(txt)

EVAL_HARNESS = r'''
import hashlib, os, time, urllib.request
if os.environ.get("STUB_BADSIG"):
    def run_episode(arm, seed):
        raise RuntimeError("wrong signature")
else:
    def run_episode(arm, seed, port, nconf, ncomp, R, log_dir):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=2)
        os.makedirs(log_dir, exist_ok=True)
        open(os.path.join(log_dir, f"f{nconf}-c{ncomp}.csv"), "a").write("row\n")
        if os.environ.get("STUB_HANG"):
            time.sleep(3600)
        time.sleep(float(os.environ.get("STUB_EP_SLOW", "0")))
        if arm.kind == "torch" and arm.broken:
            raise RuntimeError("checkpoint could not be loaded")
        if arm.kind == "torch" and str(seed) == os.environ.get("STUB_FAIL_SEED"):
            raise RuntimeError("synthetic failure")
        h = int(hashlib.md5(str(seed).encode()).hexdigest(), 16) % 60
        const = arm.kind == "const" or (arm.kind == "torch" and arm.const)
        base = {"noop": 380, "random": 100, "heur": 45}.get(arm.kind, 150)
        if arm.kind == "torch" and not const:
            base = 95 + (7 if arm.mode == "greedy" else 0)
        jitter = 0 if const else (seed * 7 + len(arm.kind)) % 13
        b = base + h - 30 + jitter
        return dict(steps=450, breaches=b, applied=90 if const else 60, reward_per_step=1.4,
                    util_mean=0.7, cpu_mean=81.0 + (h if const else 0), mem_mean=200.0 + (h if const else 0),
                    overrun_steps=0, breach_rate=b / 450, slo_used=R["slo"], freq_used=R["freq"])
'''

EVAL_ARMS = r'''
import os
class Arm:
    def __init__(self, kind, path=None, mode=None):
        self.kind, self.path, self.mode = kind, path, mode
        self.const = self.broken = False
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"checkpoint not mounted: {path}")
            txt = open(path).read().strip()
            self.const, self.broken = txt == "const", txt == "broken"
def make_arm(spec, vllm_base="http://localhost:9000"):
    kind, _, rest = spec.partition(":")
    if kind == "torch":
        path, _, mode = rest.rpartition(":")
        return Arm("torch", path, mode)
    return Arm("const" if kind in ("largest", "const") else kind)
'''

TRAINA3C = r'''
import argparse, json, os, time
p = argparse.ArgumentParser()
for f in ["-n", "-k", "-e", "-ncf", "-ncp", "-nw", "--freq", "--overrun", "--ent"]:
    p.add_argument(f)
p.add_argument("--slo"); p.add_argument("--budget", nargs=2)
a = p.parse_args()
os.makedirs(f"a3c/{a.n}/model", exist_ok=True)
json.dump({"slo": int(a.slo), "freq": int(a.freq), "budget": [int(x) for x in a.budget],
           "overrun": float(a.overrun), "knob": float(a.k), "nconf": int(a.ncf), "ncomp": int(a.ncp)},
          open(f"a3c/{a.n}/train_conditions.json", "w"))
print(f"SLO: {a.slo}, Freq: {a.freq}, Knob: {a.k}", flush=True)
if os.environ.get("STUB_TRAIN_CRASH"):
    raise ImportError("No module named 'torch_geometric'")
for i in range(1, int(a.e) + 1):
    actor = -0.02 * (1000 ** (i - 3) if os.environ.get("STUB_DIVERGE") and i > 3 else 1)
    print(f"w00 Ep: {i} | Loss:  1.5 {actor}  | Util:  0.5  | Reward:  1.3  | Ent:  0.6", flush=True)
    if i % 2 == 0:
        with open(f"a3c/{a.n}/model/model{i}.pt", "w") as f:
            f.write("learned")
    time.sleep(float(os.environ.get("STUB_TRAIN_SLOW", "0.2")))
'''

# argparse in the stub must accept the flags counselctl checks for, so the capability check passes
TRAINA3C_FLAGS = '\n# capability markers: p.add_argument("--slo") p.add_argument("--ent")\n'

FLAVORS5 = {"none": [0, 0], "2": [1, 2], "5": [2, 4], "10": [4, 8], "13": [6, 16], "19": [8, 32]}

RUNS_MD = "# Run manifest\n\n## Training runs\n\n| Run | Note |\n|---|---|\n| hand | written by hand |\n"


def sh(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def build(root):
    """Create the stub repo at root; returns its path. Tag `fix6` = training code."""
    r = os.path.join(root, "repo")
    files = {
        "tools/eval_harness.py": EVAL_HARNESS,
        "tools/eval_arms.py": EVAL_ARMS,
        "traina3c.py": TRAINA3C + TRAINA3C_FLAGS,
        "model/configs/flavors5.json": json.dumps(FLAVORS5),
        "load-gen/loadserver.py": "# stub\n",
        "load-gen/arrival_rates/2/load_150.npy": "stub\n",
        "results/RUNS.md": RUNS_MD,
        ".gitignore": "runs/\na3c/\n",
    }
    for rel, txt in files.items():
        p = os.path.join(r, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _w(p, txt)
    sh(["git", "init", "-q"], r)
    sh(["git", "config", "user.email", "test@example.com"], r)
    sh(["git", "config", "user.name", "test"], r)
    sh(["git", "add", "-A"], r)
    sh(["git", "commit", "-qm", "stub repo"], r)
    sh(["git", "tag", "fix6"], r)
    sh(["git", "tag", "eb27ae7"], r)   # loss-order marker: fix6 prints (critic, actor), as on the real repo
    # existing checkpoints (git-ignored, like the real a3c/)
    old = os.path.join(r, "a3c", "oldrun")
    os.makedirs(os.path.join(old, "model"))
    for e, txt in ((100, "const"), (200, "learned"), (300, "learned")):
        _w(os.path.join(old, "model", f"model{e}.pt"), txt)
    _w(os.path.join(old, "train_conditions.json"), json.dumps({"slo": 150, "freq": 6666, "budget": [2500, 3600],
                                                               "overrun": 0.2, "knob": 0.05, "nconf": 5, "ncomp": 3}))
    oth = os.path.join(r, "a3c", "otherslo", "model")
    os.makedirs(oth)
    _w(os.path.join(oth, "model1000.pt"), "learned")
    _w(os.path.join(r, "a3c", "otherslo", "train_conditions.json"), json.dumps({"slo": 250, "freq": 4000, "budget": [600, 800]}))
    brk = os.path.join(r, "a3c", "broken", "model")
    os.makedirs(brk)
    _w(os.path.join(brk, "model500.pt"), "broken")
    return r
