
SYSTEM = ("You configure VM instances for ONE component of a service chain. "
          "Mode ADD: choose a size to add (0 = no change). "
          "Mode REMOVE: choose a size currently held to remove (0 = no change). "
          "Keep every component's load below 1.0 and total CPU/memory within budget; "
          "do not over-provision. Reply with the option number only.")

def option_list(env, mask):
    """Flavor indices offered to the model: [0] + one representative per (cpu,mem) shape, sorted by size."""
    comp = env.components[env.act_comp]
    held = set(comp.get_instances())
    reps = {}
    for i in range(1, len(env.flavors)):
        if mask[i] != 1: continue
        if env.act_type == 0 and env.flavors[i][0] not in held: continue  # REMOVE: only held flavors
        reps.setdefault(tuple(env.flavors[i][1]), i)                     # dedupe identical shapes
    return [0] + [reps[s] for s in sorted(reps)]

def build_state_text(env, obs, options):
    mode = "ADD" if env.act_type == 1 else "REMOVE"
    arrival = float(obs[0, env.ncomp + 3])             # feature col 3 = arrival rate
    chain = "\n".join(f"- {c.name}: cpu={c.cpu} mem={c.mem} load={c.util:.2f}"
                      + (" <- ACT" if k == env.act_comp else "")
                      for k, c in enumerate(env.components))
    tcpu = sum(c.cpu for c in env.components); tmem = sum(c.mem for c in env.components)
    opts = "\n".join("0: no change" if k == 0 else
                     f"{k}: {env.flavors[i][1][0]} vCPU, {env.flavors[i][1][1]} GB"
                     for k, i in enumerate(options))
    return (f"Mode: {mode}\nArrival rate: {arrival:.1f}\nBudget (cpu, mem): {env.budget}; "
            f"used: ({tcpu}, {tmem})\nChain:\n{chain}\nOptions:\n{opts}")

def label_for(env, options, action):
    if action == 0: return 0
    s = tuple(env.flavors[action][1])
    for k in range(1, len(options)):
        if tuple(env.flavors[options[k]][1]) == s: return k
    return None                                         # action not representable -> drop sample