"""eval_arms.py -- every policy ("arm") behind one interface for eval_harness.py.

Contract
  arm.reset(seed)            called once at the start of each episode
  arm.act(env, obs, mask)    called ONLY on real decision steps (the harness forces
                             action 0 on step 0 and whenever env.act_comp == -1)
  arm.stats()                per-episode counters merged into the results row

Arm spec strings (CLI):
  noop                      always 0
  random                    uniform over legal actions (incl. 0)
  largest                   largest flavor if legal, else 0   (= rule_arm.py default)
  const:CxM                 flavor with shape (C vCPU, M GB) if legal, else 0
  heur[:grow]               obs-only right-sizing heuristic (see HeuristicArm)
  torch:PATH[:greedy|sample] any GCNActorCritic checkpoint (A3C or PPO), masked
  slm:MODEL[:PROMPT]        zero-shot LLM via vLLM; PROMPT is a key of PROMPTS
"""
import math
import numpy as np


# ---------------------------------------------------------------- simple arms
class NoOpArm:
    def reset(self, seed): pass
    def act(self, env, obs, mask): return 0
    def stats(self): return {}


class RandomArm:
    def __init__(self, offset=0):
        self.offset = offset
    def reset(self, seed):
        # policy RNG derived from the episode seed: reproducible, independent of the load RNG
        self.rng = np.random.default_rng([seed, self.offset])
    def act(self, env, obs, mask):
        return int(self.rng.choice(np.flatnonzero(np.asarray(mask) == 1)))
    def stats(self): return {}


class ConstShapeArm:
    """Always pick one shape. In REMOVE mode it is only legal if the component holds it,
    so this reproduces rule_arm.py semantics (and the collapsed A3C/PPO behaviour)."""
    def __init__(self, shape=None):
        self.shape = shape                      # None -> largest shape
    def reset(self, seed): pass
    def _index(self, env):
        if self.shape is None:
            return max(range(1, len(env.flavors)), key=lambda i: tuple(env.flavors[i][1]))
        for i in range(1, len(env.flavors)):
            if tuple(env.flavors[i][1]) == self.shape:
                return i
        raise ValueError(f"shape {self.shape} not in flavors{env.nconf}.json")
    def act(self, env, obs, mask):
        i = self._index(env)
        return i if mask[i] == 1 else 0
    def stats(self): return {}


class HeuristicArm:
    """Obs-only right-sizer (no access to the uncapped load).

    ADD: synthetic.py only raises ADD when demand > capacity/0.85 (~1.18x), and the
    obs caps util at 1, so the true shortfall is unknown. Grow capacity by a fixed
    fraction `grow` in BOTH dims: smallest shape with cpu>=grow*cpu and mem>=grow*mem
    (fallback: largest shape).
    REMOVE: raised only when cpu AND mem ratios are < 0.4, so the max per-dim ratio is
    bounded by min(0.4, sqrt(2)*util). Remove the largest held instance whose removal
    keeps that bound times the capacity shrink <= target; else no change.
    """
    def __init__(self, grow=0.5, target=0.7):
        self.grow, self.target = grow, target
    def reset(self, seed): pass
    def act(self, env, obs, mask):
        comp = env.components[env.act_comp]
        shapes = [(i, *env.flavors[i][1]) for i in range(1, len(env.flavors)) if mask[i] == 1]
        if env.act_type == 1:                                    # ADD
            ok = [s for s in shapes if s[1] >= self.grow * comp.cpu and s[2] >= self.grow * comp.mem]
            pick = min(ok, key=lambda s: (s[1], s[2])) if ok else max(shapes, key=lambda s: (s[1], s[2]))
            return pick[0]
        held = set(comp.get_instances())                         # REMOVE
        r_max = min(0.4, math.sqrt(2) * comp.util)
        for i, c, m in sorted((s for s in shapes if env.flavors[s[0]][0] in held),
                              key=lambda s: (s[1], s[2]), reverse=True):
            if comp.cpu - c <= 0 or comp.mem - m <= 0:
                continue                                         # never empty a component
            if r_max * max(comp.cpu / (comp.cpu - c), comp.mem / (comp.mem - m)) <= self.target:
                return i
        return 0
    def stats(self): return {}


# ---------------------------------------------------------------- RL checkpoints
class TorchArm:
    """Loads a pickled GCNActorCritic (A3C model/modelN.pt or a PPO model.pt).
    Masking replicates GCNActorCritic.step. 'sample' matches the authors' inference.py;
    'greedy' removes policy sampling noise."""
    def __init__(self, path, mode="sample"):
        import torch
        torch.set_num_threads(1)                                 # 4 workers x 1 thread, no oversubscription
        self.torch, self.mode = torch, mode
        try:
            self.model = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:                                        # torch < 2.6 has no weights_only
            self.model = torch.load(path, map_location="cpu")
        self.model.eval()
    def reset(self, seed):
        self.gen = self.torch.Generator().manual_seed(int(seed))
    def act(self, env, obs, mask):
        t = self.torch
        with t.no_grad():
            logits = self.model.pi._get_logits(t.as_tensor(obs, dtype=t.float32)).clone()
            logits[t.as_tensor(mask) == 0] = float("-inf")
            if self.mode == "greedy":
                return int(logits.argmax())
            return int(t.multinomial(t.softmax(logits, -1), 1, generator=self.gen))
    def stats(self): return {}


# ---------------------------------------------------------------- zero-shot SLM
from prompt_format import SYSTEM as _OPERATOR_SYSTEM

PROMPTS = {
    "operator": _OPERATOR_SYSTEM,          # the operator-goal prompt already in prompt_format.py
    # "reward": ...                        # reward-aligned prompt: next module
}


class SLMArm:
    def __init__(self, model, prompt="operator", base="http://localhost:9000", timeout=10.0):
        from llm_policy import VLLMPolicy
        if prompt not in PROMPTS:
            raise ValueError(f"unknown prompt '{prompt}', have {list(PROMPTS)}")
        self.pol = VLLMPolicy(base=base, model=model, timeout=timeout, system=PROMPTS[prompt])
    def reset(self, seed):
        # VLLMPolicy accumulates across calls; zero it so stats are per episode
        self.pol.lat, self.pol.invalid, self.pol.errors = [], 0, 0
    def act(self, env, obs, mask):
        return self.pol.get_action(env, obs, mask)
    def stats(self):
        lat = self.pol.lat
        return {"llm_queries": len(lat), "llm_invalid": self.pol.invalid,
                "llm_errors": self.pol.errors,
                "llm_lat_ms_mean": 1e3 * float(np.mean(lat)) if lat else None,
                "llm_lat_ms_p95": 1e3 * float(np.percentile(lat, 95)) if lat else None}


# ---------------------------------------------------------------- factory
def make_arm(spec, vllm_base="http://localhost:9000"):
    kind, _, rest = spec.partition(":")
    if kind == "noop":    return NoOpArm()
    if kind == "random":  return RandomArm()
    if kind == "largest": return ConstShapeArm(None)
    if kind == "const":
        c, m = rest.lower().split("x"); return ConstShapeArm((int(c), int(m)))
    if kind == "heur":    return HeuristicArm(grow=float(rest) if rest else 0.5)
    if kind == "torch":
        path, _, mode = rest.rpartition(":") if rest.endswith((":greedy", ":sample")) else (rest, "", "sample")
        return TorchArm(path, mode)
    if kind == "slm":
        model, _, prompt = rest.partition(":")
        return SLMArm(model, prompt or "operator", base=vllm_base)
    raise ValueError(f"unknown arm spec: {spec}")
