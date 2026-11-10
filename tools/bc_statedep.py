import json, sys, re, collections as C, statistics as S
rows = [json.loads(l) for l in open(sys.argv[1])]
g = C.defaultdict(list)
for r in rows:
    act = re.search(r"- (\w+):[^\n]*<- ACT", r["input"]).group(1)
    mode = "ADD" if r["meta"]["mode"] == 1 else "REM"
    g[(mode, act)].append(r["meta"]["p_top"])
for k, v in sorted(g.items()):
    print(k, "n=%d" % len(v), "p_top mean %.2f min %.2f max %.2f" % (S.mean(v), min(v), max(v)))
