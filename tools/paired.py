#!/usr/bin/env python3
"""paired.py -- paired per-seed comparison of eval_harness arms.

    python3 tools/paired.py results/eval/fix6-eval random heur fix5_m900_s e1_m300_s e1_m1000_s

Every arm ran the same seeds, so compare seed by seed: diff = arm - reference on the SAME load trace.
Prints mean diff, a 95% t-interval (n-1 df) and wins/ties/losses per pair. Stdlib only.
"""
import json, math, sys, statistics as st

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}

def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)]

def main(run, arms, metric="breaches"):
    rows = [json.loads(l) for l in open(f"{run}/episodes.jsonl")]
    by = {}
    for r in rows:
        if "error" not in r:
            by.setdefault(r["arm"], {})[r["seed"]] = r[metric]
    missing = [a for a in arms if a not in by]
    if missing:
        sys.exit(f"no rows for {missing}; arms present: {sorted(by)[:8]} ...")
    print(f"metric: {metric} (lower is better for breaches)\n")
    print(f"{'arm':<14} {'vs':<14} {'n':>3} {'mean diff':>10} {'95% CI':>18}  W/T/L")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            seeds = sorted(set(by[a]) & set(by[b]))
            d = [by[a][s] - by[b][s] for s in seeds]
            m = st.mean(d)
            h = t975(len(d) - 1) * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            w = sum(x < 0 for x in d); t = sum(x == 0 for x in d); l = sum(x > 0 for x in d)
            flag = "  *" if (m - h > 0 or m + h < 0) else ""
            print(f"{a:<14} {b:<14} {len(d):>3} {m:>10.1f} [{m - h:>7.1f}, {m + h:>7.1f}]  {w}/{t}/{l}{flag}")
    print("\n* = interval excludes 0.  W/T/L = seeds where the first arm had fewer/equal/more.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
