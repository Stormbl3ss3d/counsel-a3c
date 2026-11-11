"""bc_collect.py -- record (prompt, teacher label) pairs from ANY teacher arm.

The teacher is an eval_arms spec, so swapping teachers is a one-string change:
  torch:PATH[:greedy|sample]   an A3C/PPO checkpoint (probabilities recorded -> p_top)
  heur[:grow] | const:CxM | ... any rule arm (deterministic -> p_top = 1.0)

Episodes run through eval_harness.run_episode, unchanged: same regime, fresh env per
episode, same decision rule (step > 0 and act_comp != -1), same legality rule. The
teacher is wrapped in a Recorder, so data is collected under exactly the conditions the
student is later evaluated in. Prompts come from prompt_format/eval_arms.PROMPTS, the
same code SLMArm sends to vLLM, so train and serve prompts match by construction.

  python3 tools/bc_collect.py --name fix5m900 \
      --teacher torch:a3c/t113-fix5-f5-c3/model/model900.pt:greedy \
      --seeds 2000-2099 --ports 8000 8001 8002 8003 --nconf 5 --ncomp 3

Output (results/bc/<name>/):
  raw/s<seed>.jsonl   one row per recorded decision; written atomically per episode,
                      so a re-run skips finished seeds (resumable)
  episodes.jsonl      the teacher's own episode metrics on these seeds (breaches, ...)
  raw.jsonl           all seeds concatenated in seed order (rebuilt at the end)
  manifest.json       teacher spec, checkpoint sha256, regime, seeds, prompt, label mode
"""
import argparse, collections, hashlib, json, os, subprocess, time
import multiprocessing as mp
import numpy as np


# ------------------------------------------------------------------ teacher -> option space
def teacher_probs(arm, obs, mask):
    """Masked action distribution for torch teachers (replicates TorchArm/GCNActorCritic.step);
    None for rule arms, which are treated as deterministic."""
    model = getattr(arm, "model", None)
    if model is None:
        return None
    t = arm.torch
    with t.no_grad():
        logits = model.pi._get_logits(t.as_tensor(obs, dtype=t.float32)).clone()
        logits[t.as_tensor(mask) == 0] = float("-inf")
        return t.softmax(logits, -1).numpy()


def to_option_space(env, options, probs):
    """Sum flavor probabilities per offered option. Options keep one representative per
    (cpu, mem) shape, so duplicate-shape flavors (flavors50/100) must be merged BEFORE argmax."""
    shape2k = {tuple(env.flavors[i][1]): k for k, i in enumerate(options) if k}
    op = np.zeros(len(options))
    op[0] = float(probs[0])
    for i in range(1, len(probs)):
        k = shape2k.get(tuple(env.flavors[i][1]))
        if k is not None:
            op[k] += float(probs[i])
    return op


# ------------------------------------------------------------------ recorder (wraps any arm)
class Recorder:
    """Arm wrapper: passes the teacher's action through unchanged and records one training
    row per decision. eval_harness calls act() only on real decision steps."""

    def __init__(self, inner, system, prompt_id, label_mode="argmax", teacher_spec=""):
        from prompt_format import option_list, build_state_text, label_for
        self._option_list, self._state_text, self._label_for = option_list, build_state_text, label_for
        self.inner, self.system, self.prompt_id = inner, system, prompt_id
        self.label_mode, self.teacher_spec = label_mode, teacher_spec

    def reset(self, seed):
        self.inner.reset(seed)
        self.seed, self.k, self.rows = seed, 0, []
        self.skipped = collections.Counter()

    def act(self, env, obs, mask):
        a = int(self.inner.act(env, obs, mask))
        self.k += 1
        # the harness replaces an illegal action with 0; label what is actually applied
        a_eff = a if (0 <= a < len(mask) and mask[a] == 1) else 0
        options = self._option_list(env, mask)
        if len(options) == 1:                    # SLMArm/VLLMPolicy never queries these
            self.skipped["single_option"] += 1
            return a
        taken_k = self._label_for(env, options, a_eff)
        probs = teacher_probs(self.inner, obs, mask)
        if probs is None:                        # deterministic teacher: one-hot on its action
            if taken_k is None:
                self.skipped["unrepresentable"] += 1
                return a
            op = np.eye(len(options))[taken_k]
        else:
            op = to_option_space(env, options, probs)
        label = int(op.argmax()) if self.label_mode == "argmax" else taken_k
        if label is None:
            self.skipped["unrepresentable"] += 1
            return a
        comp = env.components[env.act_comp]
        self.rows.append({
            "instruction": self.system,          # field names kept for bc_train.py
            "input": self._state_text(env, obs, options),
            "output": str(label),
            "meta": {"episode": f"s{self.seed}", "seed": self.seed, "k": self.k,
                     "mode": int(env.act_type), "comp": comp.name, "util": float(comp.util),
                     "n_options": len(options), "label": label, "taken": taken_k,
                     "p_top": float(op.max()), "p_label": float(op[label]),
                     "probs": [round(float(x), 4) for x in op],
                     "teacher": self.teacher_spec, "prompt_id": self.prompt_id}})
        return a

    def stats(self):
        s = dict(self.inner.stats())
        s.update(bc_rows=len(self.rows), **{f"bc_skip_{k}": v for k, v in self.skipped.items()})
        return s


# ------------------------------------------------------------------ worker (one per port)
def worker(port, jobs, results, a):
    from eval_arms import make_arm, PROMPTS
    from eval_harness import run_episode, REGIMES
    rec = Recorder(make_arm(a.teacher), PROMPTS[a.prompt], a.prompt, a.label_mode, a.teacher)
    R = REGIMES[a.regime]
    while (seed := jobs.get()) is not None:
        try:
            t0 = time.time()
            ep = run_episode(rec, seed, port, a.nconf, a.ncomp, R,
                             os.path.join(a.out, "envlogs", f"p{port}"))
            final = os.path.join(a.out, "raw", f"s{seed}.jsonl")
            with open(final + ".tmp", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rec.rows)
            os.replace(final + ".tmp", final)     # atomic: a file exists only if complete
            ep.update(arm="teacher", spec=a.teacher, seed=seed, port=port, wall_s=time.time() - t0)
        except Exception as e:
            ep = dict(arm="teacher", spec=a.teacher, seed=seed, port=port, error=repr(e))
        results.put(ep)
    results.put(None)


# ------------------------------------------------------------------ provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None                              # git may be absent in the container


def teacher_checkpoint(spec):
    kind, _, rest = spec.partition(":")
    if kind != "torch":
        return None
    return rest.rpartition(":")[0] if rest.endswith((":greedy", ":sample")) else rest


def parse_seeds(s):
    if "-" in s:
        lo, hi = map(int, s.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--teacher", required=True, help="eval_arms spec, e.g. torch:PATH:greedy or heur")
    p.add_argument("--seeds", default="2000-2099", help="keep disjoint from eval seeds (1000-1009)")
    p.add_argument("--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003])
    p.add_argument("--nconf", type=int, default=5)
    p.add_argument("--ncomp", type=int, default=3)
    p.add_argument("--regime", default="eval")
    p.add_argument("--prompt", default="operator", help="key of eval_arms.PROMPTS")
    p.add_argument("--label_mode", choices=["argmax", "taken"], default="argmax")
    a = p.parse_args()
    a.out = os.path.join("results", "bc", a.name)
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)

    from eval_arms import PROMPTS
    from eval_harness import REGIMES
    seeds = parse_seeds(a.seeds)
    if set(seeds) & set(range(1000, 1010)):
        raise SystemExit("refusing: seeds overlap the eval seeds 1000-1009 (test-set leakage)")
    ckpt = teacher_checkpoint(a.teacher)
    manifest = {"name": a.name, "teacher": a.teacher,
                "teacher_sha256": sha256(ckpt) if ckpt else None,
                "label_mode": a.label_mode, "prompt_id": a.prompt,
                "prompt_sha256": hashlib.sha256(PROMPTS[a.prompt].encode()).hexdigest()[:16],
                "regime": a.regime, "regime_params": REGIMES[a.regime],
                "nconf": a.nconf, "ncomp": a.ncomp, "seeds": a.seeds,
                "git_commit": git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):                    # resuming: the data definition must not change
        old = json.load(open(mpath))
        for key in ("teacher", "teacher_sha256", "label_mode", "prompt_sha256",
                    "regime_params", "nconf", "ncomp"):
            if old.get(key) != manifest[key]:
                raise SystemExit(f"refusing to mix data: {key} differs from existing manifest "
                                 f"({old.get(key)!r} vs {manifest[key]!r}); use a new --name")
    json.dump(manifest, open(mpath, "w"), indent=1)

    todo = [s for s in seeds if not os.path.exists(os.path.join(a.out, "raw", f"s{s}.jsonl"))]
    print(f"{len(todo)} episodes to collect, {len(seeds) - len(todo)} already done, "
          f"{len(a.ports)} workers")
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    for s in todo: jobs.put(s)
    for _ in a.ports: jobs.put(None)
    procs = [ctx.Process(target=worker, args=(pt, jobs, results, a)) for pt in a.ports]
    for pr in procs: pr.start()

    alive, t0 = len(procs), time.time()
    with open(os.path.join(a.out, "episodes.jsonl"), "a") as f:
        while alive:
            r = results.get()
            if r is None: alive -= 1; continue
            f.write(json.dumps(r) + "\n"); f.flush()
            tag = f"ERROR {r['error']}" if "error" in r else \
                  f"rows={r.get('bc_rows', 0)} breaches={r['breaches']}"
            print(f"[{time.time() - t0:7.1f}s] seed={r['seed']} p{r['port']} {tag}", flush=True)
    for pr in procs: pr.join()

    labels, n = collections.Counter(), 0
    with open(os.path.join(a.out, "raw.jsonl"), "w") as out:
        for s in seeds:
            path = os.path.join(a.out, "raw", f"s{s}.jsonl")
            if not os.path.exists(path): continue
            for line in open(path):
                out.write(line); n += 1
                labels[json.loads(line)["output"]] += 1
    missing = [s for s in seeds if not os.path.exists(os.path.join(a.out, "raw", f"s{s}.jsonl"))]
    print(f"raw.jsonl: {n} rows; labels {dict(sorted(labels.items()))}; "
          f"missing seeds: {missing or 'none'}")
