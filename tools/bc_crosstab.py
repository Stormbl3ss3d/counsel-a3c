import json, sys, re, collections as C
rows = [json.loads(l) for l in open(sys.argv[1])]

def shape(r):
    k = r["output"]
    if k == "0": return "no change"
    m = re.search(rf"^{k}: (.+)$", r["input"], re.M)
    return m.group(1) if m else "?"

tab = C.Counter((("ADD" if r["meta"]["mode"] == 1 else "REMOVE"), shape(r)) for r in rows)
for (mode, s), n in sorted(tab.items()):
    print(f"{mode:7} {s:20} {n}")

print("\nfirst 30 decisions: t, mode, acting component, chosen shape, loads")
for r in rows[:30]:
    act = re.search(r"- (\w+):[^\n]*<- ACT", r["input"]).group(1)
    loads = re.findall(r"load=([\d.]+)", r["input"])
    print(r["meta"]["t"], "ADD" if r["meta"]["mode"] == 1 else "REM", act, shape(r), loads)
