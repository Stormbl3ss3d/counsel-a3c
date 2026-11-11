"""Tests for the SLM (vLLM) arms in tools/counselctl.py. Stdlib only; no Docker, no GPU, no real vLLM.
Run from the repo root:  python3 -m unittest tests.test_slm_ctl -v

The driver test runs counselctl's real EVAL_DRIVER as a subprocess against fake eval_harness /
eval_arms modules and a fake vLLM HTTP server, so the in-container path is exercised end to end.
"""
import json, os, shutil, subprocess, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import counselctl as c  # noqa: E402


def put(path, txt):
    with open(path, "w") as f:
        f.write(txt)


def spec(**over):
    u = {"name": "t1", "models": [{"label": "old", "from": "a3c/oldrun"}]}
    u.update(over)
    return u


def resolved(**over):
    return c.validate(c.merge(spec(**over)))


class FakeVLLM(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        b = json.dumps({"data": [{"id": "bc-x", "root": "/m/merged", "max_model_len": 2048}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)


class Server:
    def __enter__(self):
        self.srv = HTTPServer(("127.0.0.1", 0), FakeVLLM)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.srv.server_port}"

    def __exit__(self, *a):
        self.srv.shutdown(); self.srv.server_close()


class SpecTests(unittest.TestCase):
    def test_defaults_and_old_specs(self):
        s = resolved()
        self.assertEqual(s["slm"], [])
        self.assertEqual(s["vllm"]["base"], "http://localhost:9000")
        s = resolved(slm=[{"label": "bc", "model": "bc-heur100-l1b-e2"}])
        self.assertEqual(s["slm"][0]["prompt"], "operator")

    def test_slm_rules(self):
        cases = [([{"label": "BC", "model": "m"}], "label required"),
                 ([{"label": "bc", "model": "a:b"}], "no ':'"),
                 ([{"label": "bc"}], "served model name"),
                 ([{"label": "bc", "model": "m", "prompt": "Bad Prompt"}], "PROMPTS"),
                 ([{"label": "bc", "model": "m"}, {"label": "bc", "model": "n"}], "duplicate labels")]
        for slm, want in cases:
            with self.subTest(want=want), self.assertRaises(c.SpecError) as e:
                resolved(slm=slm)
            self.assertIn(want, str(e.exception))

    def test_unknown_slm_key_rejected(self):
        with self.assertRaises(c.SpecError) as e:
            c.merge(spec(slm=[{"label": "bc", "model": "m", "temperature": 0.7}]))
        self.assertIn("temperature", str(e.exception))

    def test_vllm_base_must_have_no_path(self):
        resolved(vllm={"base": "http://10.0.0.5:9001"})
        with self.assertRaises(c.SpecError):
            resolved(vllm={"base": "http://localhost:9000/v1"})

    def test_hf_style_model_name_allowed(self):
        s = resolved(slm=[{"label": "zs", "model": "unsloth/Llama-3.2-1B-Instruct"}])
        arms = c.arms_for("/r", s, [])
        self.assertIn(("slm_zs", "slm:unsloth/Llama-3.2-1B-Instruct:operator", None), arms)


class ArmsAndDockerTests(unittest.TestCase):
    def test_slm_arms_after_baselines_and_checkpoints(self):
        s = resolved(eval={"baselines": ["random"], "reference": ["random"], "modes": ["greedy"]},
                     slm=[{"label": "bc", "model": "bc-x"}])
        arms = c.arms_for("/r", s, [(s["models"][0], "/r/a3c/oldrun/model", [900])])
        self.assertEqual([a[0] for a in arms], ["random", "old_m900_g", "slm_bc"])
        self.assertEqual(arms[-1][1], "slm:bc-x:operator")

    def test_container_base(self):
        self.assertEqual(c.container_base("http://localhost:9000"), "http://host.docker.internal:9000")
        self.assertEqual(c.container_base("http://127.0.0.1:9000"), "http://host.docker.internal:9000")
        self.assertEqual(c.container_base("http://10.0.0.5:9000"), "http://10.0.0.5:9000")
        self.assertEqual(c.container_base("http://localhostx:9000"), "http://localhostx:9000")

    def test_docker_extra_args_only_when_given(self):
        plain = c.docker_cmd("n", "img", [], "true", "w")
        self.assertNotIn("--add-host", plain)
        with_slm = c.docker_cmd("n", "img", [], "true", "w", ["--add-host", "host.docker.internal:host-gateway"])
        i = with_slm.index("--add-host")
        self.assertLess(i, with_slm.index("img"))               # docker options must precede the image
        self.assertEqual(with_slm[i + 1], "host.docker.internal:host-gateway")

    def test_vllm_models(self):
        with Server() as base:
            self.assertEqual([m["id"] for m in c.vllm_models(base)], ["bc-x"])


FAKE_HARNESS = '''
def run_episode(arm, seed, port, nconf, ncomp, R, log_dir):
    return dict(breaches=seed % 7, applied=1, cpu_mean=1.0, mem_mean=1.0, steps=10, **arm.stats())
'''
FAKE_ARMS = '''
class Arm:
    def __init__(self, spec, base): self.spec, self.base = spec, base
    def stats(self): return {"got_base": self.base}
def make_arm(spec, vllm_base="http://localhost:9000"):
    return Arm(spec, vllm_base)
'''


class DriverTests(unittest.TestCase):
    """The real EVAL_DRIVER, run as a subprocess against fake eval_harness/eval_arms."""

    def run_driver(self, arms, vllm_base):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "tools"))
        put(os.path.join(d, "tools", "eval_harness.py"), FAKE_HARNESS)
        put(os.path.join(d, "tools", "eval_arms.py"), FAKE_ARMS)
        put(os.path.join(d, "driver.py"), c.EVAL_DRIVER)
        cfg = {"arms": arms, "seeds": [1, 2], "ports": [8000], "nconf": 5, "ncomp": 3, "envlogs": d,
               "code": d, "episodes": os.path.join(d, "episodes.jsonl"), "R": {}, "vllm_base": vllm_base}
        put(os.path.join(d, "cfg.json"), json.dumps(cfg))
        p = subprocess.run([sys.executable, os.path.join(d, "driver.py"), os.path.join(d, "cfg.json")],
                           capture_output=True, text=True, timeout=60)
        rows = [json.loads(l) for l in c._read(cfg["episodes"]).splitlines()] if os.path.exists(cfg["episodes"]) else []
        return p, rows

    def test_slm_arm_gets_container_base_after_preflight(self):
        with Server() as base:
            p, rows = self.run_driver([["random", "random"], ["slm_bc", "slm:bc-x:operator"]], base)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("vLLM reachable", p.stdout)
        self.assertEqual({r["got_base"] for r in rows if r["arm"] == "slm_bc"}, {base})
        self.assertEqual(len(rows), 4)

    def test_unreachable_vllm_stops_before_any_episode(self):
        p, rows = self.run_driver([["slm_bc", "slm:bc-x:operator"]], "http://127.0.0.1:9")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("vLLM not reachable from the container", p.stdout + p.stderr)
        self.assertEqual(rows, [])

    def test_no_slm_no_preflight_default_make_arm(self):
        p, rows = self.run_driver([["random", "random"]], None)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertNotIn("vLLM", p.stdout)
        self.assertEqual({r["got_base"] for r in rows}, {"http://localhost:9000"})


class ReportTests(unittest.TestCase):
    def build(self, errors):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        out = os.path.join(tmp, "results", "eval", "t1"); os.makedirs(out)
        s = resolved(eval={"seeds": "1-3", "modes": ["greedy"], "baselines": ["random", "const:1x2"],
                           "reference": ["random"]}, slm=[{"label": "bc", "model": "bc-x"}])
        rows = []
        for seed in (1, 2, 3):
            base = dict(seed=seed, applied=50, cpu_mean=80.0, mem_mean=200.0, reward_per_step=1.4,
                        util_mean=0.7, overrun_steps=0)
            rows += [dict(base, arm="random", breaches=100 + seed),
                     dict(base, arm="const_1x2", breaches=150 + seed, applied=90),
                     dict(base, arm="slm_bc", spec="slm:bc-x:operator", breaches=40 + seed, llm_queries=100,
                          llm_invalid=2, llm_errors=errors, llm_lat_ms_mean=20.0, llm_lat_ms_p95=40.0)]
        c._write(os.path.join(out, "episodes.jsonl"), "\n".join(map(json.dumps, rows)) + "\n")
        prov = {"eval_commit": "abc1234", "tool_describe": "x", "image_id": "sha256:feed"}
        c.build_report(tmp, s, out, [], prov, {}, [])
        return c._read(os.path.join(out, "report.md"))

    def test_slm_table(self):
        rep = self.build(errors=0)
        self.assertIn("## SLM arms (served by vLLM)", rep)
        self.assertIn("| slm_bc | `bc-x:operator` | 100.0 | 2.0% | 0.0% | 20 / 40 |", rep)
        self.assertNotIn("WARNING slm_bc", rep)

    def test_failed_requests_flagged(self):
        rep = self.build(errors=5)
        self.assertIn("WARNING slm_bc: 5.0% of LLM requests failed", rep)

    def test_no_slm_section_without_slm_arms(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        out = os.path.join(tmp, "o"); os.makedirs(out)
        s = resolved(eval={"seeds": "1-2", "modes": ["greedy"], "baselines": ["random"], "reference": ["random"]})
        c._write(os.path.join(out, "episodes.jsonl"), "\n".join(json.dumps(dict(
            arm="random", seed=k, breaches=k, applied=1, cpu_mean=1.0, mem_mean=1.0, reward_per_step=1.0,
            util_mean=0.5, overrun_steps=0)) for k in (1, 2)) + "\n")
        c.build_report(tmp, s, out, [], {"eval_commit": "abc", "tool_describe": "x", "image_id": None}, {}, [])
        self.assertNotIn("SLM arms", c._read(os.path.join(out, "report.md")))


if __name__ == "__main__":
    unittest.main()
