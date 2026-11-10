import json, csv, os, re, sys, statistics as S, collections as C
ROOT = os.environ.get("BC_ROOT", "results/raw/eval")   # holds bc_data/ and bc_logs/
run = sys.argv[1]                      # e.g. fix2  -> reads $ROOT/bc_data/fix2_m{k}.jsonl, $ROOT/bc_logs/fix2/m{k}/f25-c3.csv
cfg = sys.argv[2] if len(sys.argv) > 2 else "f25-c3"

def shape(r):
    o = r["output"]
    if o == "0": return "none"
    m = re.search(rf"^{o}: (.+)$", r["input"], re.M)
    return m.group(1) if m else "?"

print(f"{'ckpt':>5} {'dec':>4} {'p_med':>5} {'p_min':>5} {'#ADDshapes':>10} {'topADD':>14} {'#taken':>6} "
      f"{'breach':>6} {'reward':>8}  ADD p_top by component")
for k in range(100, 1001, 100):
    try:
        rows = [json.loads(l) for l in open(f"{ROOT}/bc_data/{run}_m{k}.jsonl")]
        log = list(csv.DictReader(open(f"{ROOT}/bc_logs/{run}/m{k}/{cfg}.csv")))[-1]
    except FileNotFoundError:
        print(f"{k:>5} missing"); continue
    if not rows:
        print(f"{k:>5} no decisions"); continue
    pt = [r["meta"]["p_top"] for r in rows]
    add = [r for r in rows if r["meta"]["mode"] == 1]
    shapes = C.Counter(shape(r) for r in add)
    comp = C.defaultdict(list)
    for r in add:
        a = re.search(r"- (\w+):[^\n]*<- ACT", r["input"]).group(1)
        comp[a].append(r["meta"]["p_top"])
    cs = " ".join(f"{a[:2]}={S.mean(v):.2f}" for a, v in sorted(comp.items()))
    top = shapes.most_common(1)[0][0] if shapes else "-"
    print(f"{k:>5} {len(rows):>4} {S.median(pt):>5.2f} {min(pt):>5.2f} {len(shapes):>10} {top:>14} "
          f"{len(set(r['meta']['taken'] for r in rows)):>6} {log['slo_breach']:>6} "
          f"{float(log['episode_reward']):>8.1f}  {cs}")
