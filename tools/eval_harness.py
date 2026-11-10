"""eval_harness.py -- run every arm on the same seeded episodes and write one table.

  python3 tools/eval_harness.py --run zs1 --nconf 25 --ncomp 3 --regime eval \
      --seeds 1000-1009 --ports 8000 8001 8002 8003 \
      --arms noop random largest const:1x4 heur \
             a3c_fix3=torch:a3c/fix3/model/model1000.pt:greedy \
             llama1b=slm:unsloth/Llama-3.2-1B-Instruct:operator

Output: results/eval/<run>/episodes.jsonl (one row per arm x seed, appended as jobs finish,
re-running skips rows already present), summary.csv and summary.md.
"""
import argparse, json, os, time, multiprocessing as mp
import numpy as np

REGIMES = {   # (slo, freq, knob, budget, overrun slack)
    # authors' inference.py
    "eval":  dict(slo=150, freq=int(1e6 / 150), knob=0.05, budget=[2500, 3600], slack=0.2),
    # traina3c.py regime; training draws freq = 1e6/U(200,300) per episode, fixed here at the midpoint
    "train": dict(slo=250, freq=int(1e6 / 250), knob=0.05, budget=[600, 800], slack=0.03),
}


# ------------------------------------------------------------------ one episode
def run_episode(arm, seed, port, nconf, ncomp, R, log_dir):
    from model.env import CloudEnv
    from model.synthetic import set_slo
    # Fresh env per episode: CloudEnv never resets act_comp/act_type, step_counter,
    # slo_breach or budget_overrun in reset(); a new object is the only clean state.
    # steps_per_epoch=1e9 so the load server alone ends the episode (no epoch cut).
    env = CloudEnv(log_dir, 10**9, R["budget"], R["slo"], R["slack"], "synthetic",
                   nconf, ncomp, port=port)
    set_slo(R["slo"], R["freq"], R["knob"], port=port, seed=seed)
    obs, mask = env.reset()
    arm.reset(seed)

    n = dict(steps=0, decisions=0, add_dec=0, rem_dec=0, applied=0, illegal=0,
             breaches=0, overrun_steps=0)
    reward, util, cpu, mem, lat = 0.0, [], [], [], []
    done = False
    while not done:
        decision = n["steps"] > 0 and env.act_comp != -1      # step 0: stale act state, arrival 0
        a = 0
        if decision:
            n["decisions"] += 1
            n["add_dec" if env.act_type == 1 else "rem_dec"] += 1
            t0 = time.perf_counter()
            a = int(arm.act(env, obs, mask))
            lat.append(time.perf_counter() - t0)
            if not (0 <= a < len(mask)) or mask[a] != 1:      # same legality rule for every arm
                n["illegal"] += 1; a = 0
            n["applied"] += a != 0
        obs, mask, r, done, _, _, breach = env.step(a)
        n["steps"] += 1
        reward += r
        n["breaches"] += bool(breach)
        n["overrun_steps"] += env.chain.get_budget_overrun() > R["slack"]
        util.append(np.mean([c.util for c in env.components]))   # capped at 1, as in the obs
        cpu.append(sum(c.cpu for c in env.components))
        mem.append(sum(c.mem for c in env.components))

    assert n["breaches"] == env.slo_breach, "breach count disagrees with env's own counter"
    row = dict(n, reward=reward, reward_per_step=reward / n["steps"],
               breach_rate=n["breaches"] / n["steps"],
               util_mean=float(np.mean(util)), cpu_mean=float(np.mean(cpu)),
               mem_mean=float(np.mean(mem)), cpu_max=int(max(cpu)), mem_max=int(max(mem)),
               dec_ms_mean=1e3 * float(np.mean(lat)) if lat else None,
               dec_ms_p95=1e3 * float(np.percentile(lat, 95)) if lat else None)
    row.update(arm.stats())
    return row


# ------------------------------------------------------------------ workers
def worker(port, jobs, results, a):
    """One process per load-server port: the server is stateful, so a port is never shared."""
    from eval_arms import make_arm
    arms = {}                                       # build each arm once per worker (loads ckpts once)
    R = REGIMES[a.regime]
    while (job := jobs.get()) is not None:
        name, spec, seed = job
        try:
            if name not in arms:
                arms[name] = make_arm(spec, vllm_base=a.vllm_base)
            log_dir = os.path.join(a.out, "envlogs", f"p{port}")
            t0 = time.time()
            row = run_episode(arms[name], seed, port, a.nconf, a.ncomp, R, log_dir)
            row.update(arm=name, spec=spec, seed=seed, port=port, wall_s=time.time() - t0)
        except Exception as e:                      # keep the sweep alive; record the failure
            row = dict(arm=name, spec=spec, seed=seed, port=port, error=repr(e))
        results.put(row)
    results.put(None)


# ------------------------------------------------------------------ summary
COLS = ["breaches", "breach_rate", "reward_per_step", "util_mean", "cpu_mean", "mem_mean",
        "overrun_steps", "applied", "dec_ms_mean"]

def summarise(rows, out):
    by = {}
    for r in rows:
        if "error" not in r:
            by.setdefault(r["arm"], []).append(r)
    lines, csv = [], ["arm,n," + ",".join(f"{c}_mean,{c}_std" for c in COLS) + ",invalid_rate"]
    head = "| arm | n | " + " | ".join(COLS) + " | invalid |"
    lines += [head, "|" + "---|" * (len(COLS) + 3)]
    for arm, rs in by.items():
        cells, cvals = [], []
        for c in COLS:
            v = np.array([r[c] for r in rs if r.get(c) is not None], dtype=float)
            m, s = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0) if len(v) else (np.nan, np.nan)
            cells.append(f"{m:.3g} ± {s:.2g}"); cvals += [f"{m:.6g}", f"{s:.6g}"]
        q = sum(r.get("llm_queries", 0) for r in rs)
        inv = (sum(r.get("llm_invalid", 0) + r.get("llm_errors", 0) for r in rs) / q) if q else None
        inv_s = "-" if inv is None else f"{inv:.1%}"
        lines.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + f" | {inv_s} |")
        csv.append(f"{arm},{len(rs)}," + ",".join(cvals) + f",{'' if inv is None else inv}")
    errs = [r for r in rows if "error" in r]
    if errs:
        lines.append(f"\n{len(errs)} failed episode(s), see episodes.jsonl (error field).")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(out, "summary.csv"), "w").write("\n".join(csv) + "\n")
    print("\n".join(lines))


def parse_seeds(s):
    if "-" in s:
        lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="spec or name=spec")
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--nconf", type=int, default=25); p.add_argument("--ncomp", type=int, default=3)
    p.add_argument("--regime", choices=list(REGIMES), default="eval")
    p.add_argument("--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003])
    p.add_argument("--vllm_base", default="http://localhost:9000")
    a = p.parse_args()
    a.out = os.path.join("results", "eval", a.run)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a) | {"regime_params": REGIMES[a.regime]},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)

    arms = [(x.split("=", 1) if "=" in x else (x, x)) for x in a.arms]
    path = os.path.join(a.out, "episodes.jsonl")
    old = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    have = {(r["arm"], r["seed"]) for r in old if "error" not in r}
    # seed-major order: every arm advances together, so a partial run is still a paired comparison
    todo = [(n, s, sd) for sd in parse_seeds(a.seeds) for n, s in arms if (n, sd) not in have]
    print(f"{len(todo)} episodes to run, {len(have)} already done, {len(a.ports)} workers")

    ctx = mp.get_context("spawn")                     # no inherited sockets/torch threads from the parent
    jobs, results = ctx.Queue(), ctx.Queue()
    for j in todo: jobs.put(j)
    for _ in a.ports: jobs.put(None)
    procs = [ctx.Process(target=worker, args=(pt, jobs, results, a)) for pt in a.ports]
    for pr in procs: pr.start()

    rows, alive, t0 = [r for r in old if "error" not in r], len(procs), time.time()
    with open(path, "a") as f:
        while alive:
            r = results.get()
            if r is None: alive -= 1; continue
            f.write(json.dumps(r) + "\n"); f.flush()       # survives a crash / shutdown mid-sweep
            rows.append(r)
            tag = f"ERROR {r['error']}" if "error" in r else f"breaches={r['breaches']} rwd/step={r['reward_per_step']:.3f}"
            print(f"[{time.time()-t0:7.1f}s] {r['arm']:<14} seed={r['seed']} p{r['port']} {tag}", flush=True)
    for pr in procs: pr.join()
    summarise(rows, a.out)
