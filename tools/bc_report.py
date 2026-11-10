import json, sys, csv, collections as C, statistics as S
rows = [json.loads(l) for l in open(sys.argv[1])]; m = [r["meta"] for r in rows]
print("decision steps:", len(rows))
print("label hist    :", dict(sorted(C.Counter(r["output"] for r in rows).items())))
print("taken hist    :", dict(sorted(C.Counter(str(x["taken"]) for x in m).items())))
print("mode (1=ADD)  :", dict(C.Counter(x["mode"] for x in m)))
print("p_top         : median %.2f, min %.2f" % (S.median(x["p_top"] for x in m), min(x["p_top"] for x in m)))
for r in csv.DictReader(open(sys.argv[2])):
    print("env log       :", {k: r[k] for k in ("steps","actions","episode_reward","start_util","end_util","slo_breach","budget_overrun")})