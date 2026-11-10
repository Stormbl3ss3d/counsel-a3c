
import argparse, json, collections, numpy as np, torch
from torch.distributions.categorical import Categorical
from model.env import CloudEnv
from model.synthetic import set_slo
from prompt_format import SYSTEM, option_list, build_state_text, label_for

def load_teacher(path):
    try:  m = torch.load(path, map_location="cpu", weights_only=False)  # torch>=2.6 needs this
    except TypeError: m = torch.load(path, map_location="cpu")
    return m.eval()

@torch.no_grad()
def masked_probs(model, obs, mask):                    # replicates model.step's masking
    logits = model.pi._get_logits(torch.as_tensor(obs, dtype=torch.float32)).clone()
    logits[torch.as_tensor(mask) == 0] = float("-inf")
    return torch.softmax(logits, -1)

class TrajectoryLogger:
    def __init__(self, path, label_mode="argmax", min_ptop=0.0):
        self.f = open(path, "a"); self.mode = label_mode; self.min_ptop = min_ptop
        self.stats, self.labels = collections.Counter(), collections.Counter()

    def log(self, env, obs, mask, probs, taken, episode, t):
        if env.act_comp == -1:                         # nothing flagged: no decision to learn
            self.stats["no_decision"] += 1; return
        options = option_list(env, mask)
        # sum probability over duplicate-shape flavors BEFORE argmax (f50/f100 have duplicates)
        shape2k = {tuple(env.flavors[i][1]): k for k, i in enumerate(options) if k}
        op = np.zeros(len(options)); op[0] = float(probs[0])
        for i in range(1, len(probs)):
            k = shape2k.get(tuple(env.flavors[i][1]))
            if k is not None: op[k] += float(probs[i])
        taken_k = label_for(env, options, taken)
        label = int(op.argmax()) if self.mode == "argmax" else taken_k
        if label is None or op.max() < self.min_ptop:
            self.stats["dropped"] += 1; return
        self.f.write(json.dumps({"instruction": SYSTEM,
            "input": build_state_text(env, obs, options), "output": str(label),
            "meta": {"episode": episode, "t": t, "p_top": float(op.max()),
                     "taken": taken_k, "n_options": len(options), "mode": int(env.act_type)}}) + "\n")
        self.labels[label] += 1; self.stats["logged"] += 1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True); p.add_argument("--out", required=True)
    p.add_argument("--nconf", type=int, default=5); p.add_argument("--ncomp", type=int, default=3)
    p.add_argument("--port", type=int, default=8000); p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--slo", type=int, default=150); p.add_argument("--knob", type=float, default=0.05)
    p.add_argument("--label_mode", default="argmax"); p.add_argument("--min_ptop", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_dir", default="./bc_logs")
    p.add_argument("--budget", type=int, nargs=2, default=[2500, 3600])
    a = p.parse_args()
    set_slo(a.slo, int(1e6 / a.slo), a.knob, port=a.port,seed=a.seed)
    env = CloudEnv(a.log_dir, 1000, a.budget, a.slo, 0.2, "synthetic", a.nconf, a.ncomp, port=a.port)
    teacher, lg = load_teacher(a.teacher), TrajectoryLogger(a.out, a.label_mode, a.min_ptop)
    obs, mask = env.reset(); ep = t = ep_t = 0
    while ep < a.episodes:
        probs = masked_probs(teacher, obs, mask)
        taken = int(Categorical(probs=probs).sample())
        if ep_t > 0:        # step 0 after reset has stale act state and arrival 0.0: not a real decision
            lg.log(env, obs, mask, probs.numpy(), taken, f"p{a.port}-e{ep}", t)
        obs, mask, reward, done, *_ = env.step(taken)
        t += 1; ep_t += 1
        if done: ep += 1; ep_t = 0; obs, mask = env.reset()
    lg.f.close(); print(dict(lg.stats), "label histogram:", dict(sorted(lg.labels.items())))