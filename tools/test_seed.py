# test_seed.py -- run with a load server up and no other client talking to it
import sys, requests
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
URL, N = f"http://localhost:{PORT}", 3000      # 3000 steps spans several episodes
s = requests.Session()

def trace(seed):
    s.put(f"{URL}/slo", json={"slo":150, "freq":int(1e6/150), "knob":0.05, "seed":seed}).raise_for_status()
    s.put(f"{URL}/base", json={"cpu":[4,4,4], "mem":[8,8,8]}).raise_for_status()  # /load needs a base set
    out = []
    for _ in range(N):
        m = s.get(f"{URL}/load").json()
        out.append((round(m["arrival_rate"], 6), m["done"]))
    return out

a, b, c, d, e = trace(7), trace(7), trace(8), trace(None), trace(None)
print("same seed -> identical :", a == b, "  (expect True)")
print("different seed differs :", a != c, "  (expect True)")
print("unseeded runs differ   :", d != e, "  (expect True)")
print("episodes in 3000 steps :", sum(x[1] for x in a))