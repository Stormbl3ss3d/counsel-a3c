# rule_arm.py -- "largest flavor if legal, else no change" (mimics PPO's observed behaviour)
import argparse
from model.env import CloudEnv
from model.synthetic import set_slo
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--nconf", type=int, default=5); p.add_argument("--ncomp", type=int, default=3)
p.add_argument("--port", type=int, default=8000); p.add_argument("--seed", type=int, default=None)
p.add_argument("--episodes", type=int, default=1); p.add_argument("--slo", type=int, default=150)
p.add_argument("--knob", type=float, default=0.05)
p.add_argument("--budget", type=int, nargs=2, default=[2500, 3600])
p.add_argument("--log_dir", default="./bc_logs/rule")
p.add_argument("--random", action="store_true")
p.add_argument("--rseed", type=int, default=0)
a = p.parse_args()

set_slo(a.slo, int(1e6 / a.slo), a.knob, port=a.port, seed=a.seed)
env = CloudEnv(a.log_dir, 1000, a.budget, a.slo, 0.2, "synthetic", a.nconf, a.ncomp, port=a.port)
top = max(range(1, len(env.flavors)), key=lambda i: tuple(env.flavors[i][1]))   # largest shape
obs, mask = env.reset(); ep = 0
rng = np.random.default_rng(a.rseed)
while ep < a.episodes:
    if a.random:
        act = int(rng.choice(np.flatnonzero(np.asarray(mask) == 1)))   # uniform over legal actions, every step
    else:
        act = top if mask[top] == 1 else 0       # applied on every step, as PPO did
    obs, mask, reward, done, *_ = env.step(act)
    if done: ep += 1; obs, mask = env.reset()
print("largest flavor index:", top)