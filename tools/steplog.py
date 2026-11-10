"""
steplog.py - per-step logger for COUNSEL / counsel-a3c inference.

Records, for every env step: latency (as returned by the load server model),
latency/SLO, mean / max component utilization, reward, SLO breach, budget
overrun, arrival rate, and the action taken. Writes one CSV per run.

No repo file is modified. env.py does `from .synthetic import call_load_server`,
so the name lives in model.env's namespace; we wrap model.env.call_load_server
to capture the latency that env.step() computes but never returns.

Run from the repo root (inside the tools container), e.g.:
  python3 tools/steplog.py --policy torch:a3c/std-f5-c3/model/model1000.pt \
      --nconf 5 --ncomp 3 --slo 150 --knob 0.05 --seed 7 --episodes 1 --port 8000
  python3 tools/steplog.py --policy noop --seed 7 --out results/raw/steplog/noop_s7.csv

Policies: noop | random | torch:PATH[:greedy]
  torch:PATH        -> model.step(obs, mask) (samples, like inference_seq_a3c.py)
  torch:PATH:greedy -> argmax of masked logits via net.pi._get_logits (falls
                       back to sampling if that method is missing)
"""
import argparse
import csv
import os
import sys

import numpy as np

import model.env as env_mod
from model.env import CloudEnv
from model.synthetic import set_slo

# ---------------------------------------------------------------- capture hook
_LAST = {}
_orig_call = env_mod.call_load_server


def _wrapped_call(cpu, mem, port=8000):
    out = _orig_call(cpu, mem, port=port)
    # out = (arrival_rate, loadc, loadm, latency, act_type, act_comp, done)
    _LAST.update(sent_cpu=list(cpu), sent_mem=list(mem),
                 arrival_rate=float(out[0]), latency=float(out[3]))
    return out


env_mod.call_load_server = _wrapped_call


# ---------------------------------------------------------------- policies
def make_policy(spec, n_actions, rng):
    if spec == "noop":
        return lambda o, m: 0
    if spec == "random":
        def act(o, m):
            valid = np.flatnonzero(np.asarray(m) > 0)
            return int(rng.choice(valid)) if len(valid) else 0
        return act
    if spec.startswith("torch:"):
        import torch
        parts = spec.split(":")
        path, greedy = parts[1], (len(parts) > 2 and parts[2] == "greedy")
        net = torch.load(path, map_location="cpu")
        net.eval()

        def act(o, m):
            with torch.no_grad():
                ot = torch.as_tensor(o, dtype=torch.float32)
                mt = torch.as_tensor(m, dtype=torch.float32)
                if greedy and hasattr(net.pi, "_get_logits"):
                    # same masking as GCNActorCritic.step(), then argmax
                    logits = net.pi._get_logits(ot).clone()
                    logits[mt == 0] = float("-inf")
                    return int(torch.argmax(logits).item())
                a, _, _ = net.step(ot, mt)
                return int(np.asarray(a).item())
        return act
    raise ValueError(f"unknown policy {spec!r}")


# ---------------------------------------------------------------- main
FIELDS = ["episode", "step", "action", "act_type", "act_comp",
          "arrival_rate", "latency", "lat_over_slo",
          "util_mean", "util_max", "reward", "breach", "budget_overrun",
          "sent_cpu", "sent_mem"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="noop")
    p.add_argument("--nconf", type=int, default=5)
    p.add_argument("--ncomp", type=int, default=3)
    p.add_argument("--slo", type=int, default=150)
    p.add_argument("--knob", type=float, default=0.05)
    p.add_argument("--budget", type=int, nargs=2, default=[2500, 3600])
    p.add_argument("--overrun_lim", type=float, default=0.2)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=None,
                   help="load-server seed (needs your seeded loadserver)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    freq = int(1e6 / a.slo)
    set_slo(a.slo, freq, a.knob, port=a.port, seed=a.seed)

    tag = a.policy.split(":")[0] if not a.policy.startswith("torch:") \
        else "torch_" + os.path.basename(a.policy.split(":")[1]).split(".")[0]
    out = a.out or f"results/raw/steplog/{tag}_f{a.nconf}-c{a.ncomp}_slo{a.slo}_s{a.seed}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    env = CloudEnv(os.path.join(os.path.dirname(out) or ".", "env_logs"),
                   a.steps_per_epoch, a.budget, a.slo, a.overrun_lim,
                   "synthetic", a.nconf, a.ncomp, port=a.port)
    act = make_policy(a.policy, env.action_space.n,
                      np.random.default_rng(a.seed))

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for ep in range(a.episodes):
            obs, mask = env.reset()
            done, t = False, 0
            while not done:
                act_type, act_comp = env.act_type, env.act_comp  # pre-step
                action = act(obs, mask)
                obs, mask, r, done, _info, _ul, breach = env.step(action)
                utils = [c.util for c in env.components]
                lat = _LAST["latency"]
                w.writerow(dict(
                    episode=ep, step=t, action=action,
                    act_type=act_type, act_comp=act_comp,
                    arrival_rate=_LAST["arrival_rate"],
                    latency=lat, lat_over_slo=lat / a.slo,
                    util_mean=float(np.mean(utils)),
                    util_max=float(np.max(utils)),
                    reward=float(r), breach=int(breach),
                    budget_overrun=env.chain.get_budget_overrun(),
                    sent_cpu=" ".join(map(str, _LAST["sent_cpu"])),
                    sent_mem=" ".join(map(str, _LAST["sent_mem"])),
                ))
                t += 1
            print(f"episode {ep}: {t} steps", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
