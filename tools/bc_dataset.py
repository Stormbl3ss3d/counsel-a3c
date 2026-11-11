"""bc_dataset.py -- split collected BC data by episode and compute the baselines a clone must beat.

Teacher-agnostic: reads only results/bc/<name>/raw.jsonl (+ manifest.json) written by
bc_collect.py, whatever the teacher was. Standard library only, so it runs on the host or
in the container.

  python3 tools/bc_dataset.py --name fix5m900 [--val 0.1 --test 0.1 --split_seed 0]

Output (results/bc/<name>/data/):
  train.jsonl, val.jsonl, test.jsonl   rows unchanged, split by WHOLE episode (seed), so no
                                       decision from a test trace is ever seen in training
  dataset.json                         split, label histograms, p_top histogram, baselines,
                                       warnings, source manifest and raw.jsonl sha256

Baselines: for each feature set (global / mode / mode+comp / mode+comp+loadbin /
mode+comp+n_options) a majority-label rule is FIT ON TRAIN and scored on TEST, on all rows
and on confident rows (p_top >= --confident). Unseen test groups back off to the coarser
rule. A student's offline accuracy means something only relative to these numbers.
"""
import argparse, collections, hashlib, json, os, random

FEATURE_SETS = ["global", "mode", "mode+comp", "mode+comp+loadbin", "mode+comp+n_options"]


def load_bin(u):
    return "lt0.5" if u < 0.5 else "lt1" if u < 1.0 else "ge1"


def features(meta):
    """Feature tuples per feature set, from the structured meta bc_collect writes."""
    m = "ADD" if meta["mode"] == 1 else "REM"
    c = meta["comp"]
    return {"global": (), "mode": (m,), "mode+comp": (m, c),
            "mode+comp+loadbin": (m, c, load_bin(meta["util"])),
            "mode+comp+n_options": (m, c, meta["n_options"])}


class MajorityRule:
    """Most common label per feature group; unseen groups back off through `chain`."""

    def __init__(self, chain):
        self.chain = chain                       # finest first, e.g. ["mode+comp", "mode", "global"]
        self.tables = {}

    def fit(self, rows):
        for fs in self.chain:
            g = collections.defaultdict(collections.Counter)
            for r in rows:
                g[features(r["meta"])[fs]][r["output"]] += 1
            # tie-break on the label string so the rule is deterministic
            self.tables[fs] = {k: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                               for k, c in g.items()}
        return self

    def predict(self, meta):
        f = features(meta)
        for fs in self.chain:
            if f[fs] in self.tables[fs]:
                return self.tables[fs][f[fs]]
        return None


def backoff_chain(fs):
    order = {"global": ["global"], "mode": ["mode", "global"],
             "mode+comp": ["mode+comp", "mode", "global"],
             "mode+comp+loadbin": ["mode+comp+loadbin", "mode+comp", "mode", "global"],
             "mode+comp+n_options": ["mode+comp+n_options", "mode+comp", "mode", "global"]}
    return order[fs]


def accuracy(rule, rows):
    if not rows:
        return None
    return sum(rule.predict(r["meta"]) == r["output"] for r in rows) / len(rows)


def split_episodes(episodes, val, test, seed):
    eps = sorted(episodes)
    if len(eps) < 3:
        raise SystemExit(f"need at least 3 episodes to split, have {len(eps)}")
    random.Random(seed).shuffle(eps)
    n_test = max(1, round(test * len(eps)))
    n_val = max(1, round(val * len(eps)))
    if n_test + n_val >= len(eps):
        raise SystemExit(f"{len(eps)} episodes are too few for val={val} test={test}")
    return {"test": sorted(eps[:n_test]), "val": sorted(eps[n_test:n_test + n_val]),
            "train": sorted(eps[n_test + n_val:])}


def histogram(values, edges):
    out = collections.Counter()
    for v in values:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= v < hi or (hi == edges[-1] and v == hi):
                out[f"{lo:.1f}-{hi:.1f}"] += 1
                break
    return dict(sorted(out.items()))


def build(rows, val, test, split_seed, confident, train_min_ptop):
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["meta"]["episode"]].append(r)
    split = split_episodes(by_ep, val, test, split_seed)
    parts = {k: [r for ep in eps for r in by_ep[ep]] for k, eps in split.items()}
    dropped = 0
    if train_min_ptop > 0:                       # only ever filter TRAIN; val/test stay complete
        keep = [r for r in parts["train"] if r["meta"]["p_top"] >= train_min_ptop]
        dropped = len(parts["train"]) - len(keep)
        parts["train"] = keep
    test_conf = [r for r in parts["test"] if r["meta"]["p_top"] >= confident]

    baselines = {}
    for fs in FEATURE_SETS:
        rule = MajorityRule(backoff_chain(fs)).fit(parts["train"])
        baselines[fs] = {"test_acc": accuracy(rule, parts["test"]),
                         "test_acc_confident": accuracy(rule, test_conf),
                         "train_acc": accuracy(rule, parts["train"]),
                         "groups": len(rule.tables[fs])}

    ptops = [r["meta"]["p_top"] for r in rows]
    stats = {
        "episodes": {k: len(v) for k, v in split.items()},
        "rows": {k: len(v) for k, v in parts.items()},
        "train_dropped_low_ptop": dropped,
        "test_confident_rows": len(test_conf),
        "labels": {k: dict(sorted(collections.Counter(r["output"] for r in v).items()))
                   for k, v in parts.items()},
        "modes": dict(collections.Counter("ADD" if r["meta"]["mode"] == 1 else "REM" for r in rows)),
        "p_top_hist": histogram(ptops, [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "confident_fraction": sum(p >= confident for p in ptops) / len(ptops),
    }
    return split, parts, baselines, stats


def warnings_for(baselines, stats, confident):
    w = []
    best_fs, best = max(((fs, b["test_acc"]) for fs, b in baselines.items()), key=lambda x: x[1])
    if best >= 0.95:
        w.append(f"teacher is ~a lookup table: '{best_fs}' majority rule gets {best:.1%} on test; "
                 "high clone accuracy will say little about the student")
    if stats["confident_fraction"] < 0.5:
        w.append(f"only {stats['confident_fraction']:.0%} of rows have p_top >= {confident}; "
                 "argmax labels are noisy, report confident-row accuracy alongside all-row accuracy")
    if stats["rows"]["test"] < 200:
        w.append(f"test set has {stats['rows']['test']} rows; accuracy differences under ~5 points "
                 "are within noise, collect more episodes before comparing students")
    for k, hist in stats["labels"].items():
        if k != "train" and set(hist) - set(stats["labels"]["train"]):
            w.append(f"{k} contains labels never seen in train: {sorted(set(hist) - set(stats['labels']['train']))}")
    return w


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--root", default=os.path.join("results", "bc"))
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--confident", type=float, default=0.7, help="p_top threshold for 'confident' rows")
    p.add_argument("--train_min_ptop", type=float, default=0.0,
                   help="drop train rows below this teacher confidence (val/test never filtered)")
    a = p.parse_args(argv)

    base = os.path.join(a.root, a.name)
    raw = os.path.join(base, "raw.jsonl")
    if not os.path.exists(raw):
        raise SystemExit(f"{raw} not found; run bc_collect.py --name {a.name} first")
    rows = [json.loads(l) for l in open(raw) if l.strip()]
    if not rows:
        raise SystemExit(f"{raw} is empty")
    split, parts, baselines, stats = build(rows, a.val, a.test, a.split_seed,
                                           a.confident, a.train_min_ptop)
    warns = warnings_for(baselines, stats, a.confident)

    out = os.path.join(base, "data")
    os.makedirs(out, exist_ok=True)
    for k, rs in parts.items():
        with open(os.path.join(out, f"{k}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    mpath = os.path.join(base, "manifest.json")
    info = {"name": a.name, "params": {k: v for k, v in vars(a).items() if k != "root"},
            "source_manifest": json.load(open(mpath)) if os.path.exists(mpath) else None,
            "raw_sha256": sha256_file(raw), "split": split, "stats": stats,
            "baselines": baselines, "warnings": warns}
    json.dump(info, open(os.path.join(out, "dataset.json"), "w"), indent=1)

    print(f"episodes {stats['episodes']}  rows {stats['rows']}  confident(test) {stats['test_confident_rows']}")
    print(f"labels(train) {stats['labels']['train']}   p_top {stats['p_top_hist']}")
    print("baseline (majority rule fit on train)   test_acc   test_acc_confident")
    for fs, b in baselines.items():
        conf = "-" if b["test_acc_confident"] is None else f"{b['test_acc_confident']:.3f}"
        print(f"  {fs:22} groups={b['groups']:3}   {b['test_acc']:.3f}      {conf}")
    for w in warns:
        print("WARNING:", w)
    return info


if __name__ == "__main__":
    main()
