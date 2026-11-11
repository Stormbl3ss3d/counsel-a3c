"""Tests for the BC pipeline: tools/bc_collect.py (Recorder) and tools/bc_dataset.py.

bc_dataset tests need only the standard library. Recorder tests need numpy + torch, so run the
whole file in the tools container (it has both), from the repo root:
    docker compose run --rm -T --no-deps tools python3 -m unittest tests.test_bc -v
On a host without numpy/torch the Recorder tests are skipped, not failed.
"""
import contextlib, io, json, os, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import bc_dataset as D  # noqa: E402
import bc_offline_eval as O  # noqa: E402

try:
    import numpy as np
    import torch
    import bc_collect as C
    import prompt_format as PF
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ---------------------------------------------------------------- fakes for the Recorder
if HAVE_TORCH:
    class Comp:
        def __init__(s, name, cpu, mem, util, inst):
            s.name, s.cpu, s.mem, s.util, s.inst = name, cpu, mem, util, inst
        def get_instances(s): return s.inst

    class Env:
        """flavors5-like; index 5 duplicates the shape of index 1 to test probability merging."""
        flavors = [("none", [0, 0]), ("f1", [1, 2]), ("f2", [2, 4]), ("f3", [6, 16]),
                   ("f4", [8, 32]), ("f5", [1, 2])]
        ncomp, budget = 3, [2500, 3600]
        def __init__(s, act_type, act_comp=0):
            s.act_type, s.act_comp = act_type, act_comp
            s.components = [Comp("Database", 4, 8, 1.0, ["f1", "f3"]),
                            Comp("Firewall", 2, 4, 0.6, ["f1"]),
                            Comp("WebServer", 2, 4, 0.9, ["f2"])]

    class ConstArm:
        def __init__(s, a): s.a = a
        def reset(s, seed): pass
        def act(s, env, obs, mask): return s.a
        def stats(s): return {"inner": 1}

    class FakeTorchArm:
        """Mimics eval_arms.TorchArm: exposes .model.pi._get_logits and .torch."""
        def __init__(s, logits, action=3):
            s.torch, s.action = torch, action
            class Pi:
                def _get_logits(_, o): return torch.tensor(logits, dtype=torch.float32)
            class M: pi = Pi()
            s.model = M()
        def reset(s, seed): pass
        def act(s, env, obs, mask): return s.action
        def stats(s): return {}


@unittest.skipUnless(HAVE_TORCH, "needs numpy + torch (run in the tools container)")
class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.zeros((3, 8)); self.obs[0, 6] = 123.0      # arrival column = ncomp + 3

    def rec(self, arm, mode="argmax"):
        r = C.Recorder(arm, PF.SYSTEM, "operator", mode, "teacher-spec"); r.reset(7); return r

    def test_rule_teacher_one_hot_and_prompt_matches_serving(self):
        r = self.rec(ConstArm(3))
        self.assertEqual(r.act(Env(1), self.obs, np.ones(6)), 3)    # action passes through
        row = r.rows[0]
        self.assertEqual(row["output"], "3")
        self.assertEqual(row["meta"]["p_top"], 1.0)
        self.assertEqual(row["meta"]["episode"], "s7")
        self.assertEqual(row["instruction"], PF.SYSTEM)
        env = Env(1)                                                  # same text SLMArm would send
        self.assertEqual(row["input"], PF.build_state_text(env, self.obs, PF.option_list(env, np.ones(6))))

    def test_illegal_action_labelled_as_applied_noop(self):
        r = self.rec(ConstArm(4)); mask = np.ones(6); mask[4] = 0
        self.assertEqual(r.act(Env(1), self.obs, mask), 4)
        self.assertEqual(r.rows[0]["output"], "0")

    def test_remove_mode_option_indexing(self):
        r = self.rec(ConstArm(3))                                     # remove 6x16; options [0, 1x2, 6x16]
        r.act(Env(0), self.obs, np.array([1, 1, 0, 1, 0, 1.]))
        self.assertEqual(r.rows[0]["output"], "2")
        self.assertEqual(r.rows[0]["meta"]["n_options"], 3)

    def test_single_option_state_skipped(self):
        r = self.rec(ConstArm(0))
        r.act(Env(0), self.obs, np.array([1, 0, 0, 0, 0, 0.]))
        self.assertEqual(r.rows, [])
        self.assertEqual(r.stats()["bc_skip_single_option"], 1)

    def test_duplicate_shapes_merged_before_argmax(self):
        # f1 0.3 + f5 (same shape) 0.3 = 0.6 beats f3 0.4 -> label 1, though the teacher took f3
        logits = np.log(np.array([1e-9, 0.3, 1e-9, 0.4, 1e-9, 0.3]))
        r = self.rec(FakeTorchArm(logits))
        r.act(Env(1), self.obs, np.ones(6))
        m = r.rows[0]["meta"]
        self.assertEqual(r.rows[0]["output"], "1")
        self.assertAlmostEqual(m["p_top"], 0.6, places=3)
        self.assertEqual(m["taken"], 3)

    def test_taken_label_mode(self):
        logits = np.log(np.array([1e-9, 0.3, 1e-9, 0.4, 1e-9, 0.3]))
        r = self.rec(FakeTorchArm(logits), mode="taken")
        r.act(Env(1), self.obs, np.ones(6))
        self.assertEqual(r.rows[0]["output"], "3")

    def test_masked_flavors_get_zero_probability(self):
        logits = np.zeros(6)                                          # uniform before masking
        r = self.rec(FakeTorchArm(logits, action=1))
        r.act(Env(0), self.obs, np.array([1, 1, 0, 1, 0, 1.]))       # REMOVE: f2, f4 masked
        self.assertAlmostEqual(sum(r.rows[0]["meta"]["probs"]), 1.0, places=3)

    def test_stats_merge_inner_and_rows_are_json(self):
        r = self.rec(ConstArm(1)); r.act(Env(1), self.obs, np.ones(6))
        self.assertEqual(r.stats(), {"inner": 1, "bc_rows": 1})
        json.dumps(r.rows)

    def test_reset_clears_rows(self):
        r = self.rec(ConstArm(1)); r.act(Env(1), self.obs, np.ones(6)); r.reset(8)
        self.assertEqual(r.rows, [])

    def test_teacher_checkpoint_parsing(self):
        self.assertEqual(C.teacher_checkpoint("torch:a/b/model9.pt:greedy"), "a/b/model9.pt")
        self.assertEqual(C.teacher_checkpoint("torch:a/b/model9.pt"), "a/b/model9.pt")
        self.assertIsNone(C.teacher_checkpoint("heur"))


# ---------------------------------------------------------------- bc_dataset (stdlib only)
def row(ep, label, mode=1, comp="Database", util=1.0, n_options=5, p_top=1.0):
    return {"instruction": "s", "input": "x", "output": str(label),
            "meta": {"episode": ep, "mode": mode, "comp": comp, "util": util,
                     "n_options": n_options, "p_top": p_top}}


class DatasetTests(unittest.TestCase):
    def rows(self, n_eps=10, per=10):
        out = []
        for e in range(n_eps):
            for k in range(per):
                # ADD -> label 1, REMOVE -> label 0: a pure mode lookup table
                out.append(row(f"s{e}", 1 if k % 2 else 0, mode=k % 2, p_top=0.6 + 0.04 * k))
        return out

    def test_split_is_by_episode_disjoint_complete_and_deterministic(self):
        rows = self.rows()
        s1, parts, _, _ = D.build(rows, 0.1, 0.2, 0, 0.7, 0.0)
        s2, _, _, _ = D.build(rows, 0.1, 0.2, 0, 0.7, 0.0)
        self.assertEqual(s1, s2)
        eps = [set(v) for v in s1.values()]
        self.assertFalse(eps[0] & eps[1] or eps[0] & eps[2] or eps[1] & eps[2])
        self.assertEqual(sum(len(v) for v in parts.values()), len(rows))
        self.assertEqual((len(s1["test"]), len(s1["val"])), (2, 1))
        for k, rs in parts.items():
            self.assertTrue(all(r["meta"]["episode"] in s1[k] for r in rs))

    def test_different_split_seed_changes_split(self):
        rows = self.rows(n_eps=20)
        self.assertNotEqual(D.build(rows, .1, .1, 0, .7, 0)[0], D.build(rows, .1, .1, 1, .7, 0)[0])

    def test_too_few_episodes_refused(self):
        with self.assertRaises(SystemExit):
            D.build(self.rows(n_eps=2), 0.1, 0.1, 0, 0.7, 0.0)

    def test_baselines_fit_on_train_scored_on_test(self):
        # train eps say label 1, test eps say label 2: an in-sample rule would look perfect
        rows = [row(f"s{e}", 1) for e in range(8) for _ in range(5)] + \
               [row(f"s{e}", 2) for e in range(8, 10) for _ in range(5)]
        split = {"train": [f"s{e}" for e in range(8)], "val": ["s8"], "test": ["s9"]}
        orig = D.split_episodes
        D.split_episodes = lambda *a, **k: split
        try:
            _, _, b, _ = D.build(rows, .1, .1, 0, .7, 0)
        finally:
            D.split_episodes = orig
        self.assertEqual(b["global"]["train_acc"], 1.0)
        self.assertEqual(b["global"]["test_acc"], 0.0)

    def test_mode_lookup_teacher_detected(self):
        _, _, b, stats = D.build(self.rows(), .1, .1, 0, .7, 0)
        self.assertEqual(b["mode"]["test_acc"], 1.0)
        self.assertLess(b["global"]["test_acc"], 1.0)
        self.assertTrue(any("lookup table" in w for w in D.warnings_for(b, stats, .7)))

    def test_unseen_group_backs_off(self):
        rule = D.MajorityRule(D.backoff_chain("mode+comp")).fit([row("s0", 3, comp="Firewall")])
        unseen_comp = row("t", 0, mode=1, comp="WebServer")["meta"]
        unseen_mode = row("t", 0, mode=0, comp="WebServer")["meta"]
        self.assertEqual(rule.predict(unseen_comp), "3")                          # -> mode
        self.assertEqual(rule.predict(unseen_mode), "3")                          # -> global

    def test_train_min_ptop_filters_train_only(self):
        rows = self.rows()
        _, full, _, _ = D.build(rows, .1, .1, 0, .7, 0.0)
        _, filt, _, st = D.build(rows, .1, .1, 0, .7, 0.8)
        self.assertLess(len(filt["train"]), len(full["train"]))
        self.assertEqual(len(filt["test"]), len(full["test"]))
        self.assertEqual(len(filt["val"]), len(full["val"]))
        self.assertGreater(st["train_dropped_low_ptop"], 0)

    def test_deterministic_teacher_has_all_rows_confident(self):
        rows = [row(f"s{e}", e % 3) for e in range(10) for _ in range(3)]
        _, _, _, st = D.build(rows, .1, .1, 0, .7, 0)
        self.assertEqual(st["confident_fraction"], 1.0)
        self.assertEqual(st["p_top_hist"], {"0.9-1.0": 30})

    def test_main_writes_files_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "x"); os.makedirs(base)
            with open(os.path.join(base, "raw.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in self.rows())
            json.dump({"teacher": "heur"}, open(os.path.join(base, "manifest.json"), "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                info = D.main(["--name", "x", "--root", d])
            for k in ("train", "val", "test"):
                n = sum(1 for _ in open(os.path.join(base, "data", f"{k}.jsonl")))
                self.assertEqual(n, info["stats"]["rows"][k])
            saved = json.load(open(os.path.join(base, "data", "dataset.json")))
            self.assertEqual(saved["source_manifest"]["teacher"], "heur")
            self.assertEqual(len(saved["raw_sha256"]), 64)

    def test_missing_raw_refused(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            D.main(["--name", "nope", "--root", d])


# ---------------------------------------------------------------- bc_offline_eval (fake vLLM)
class FakeVLLM(BaseHTTPRequestHandler):
    """Serves /v1/models and /v1/chat/completions. The reply is chosen by the user text:
    'ANS=<text>' -> that text, 'FAIL' -> HTTP 500. Records every request body."""
    bodies = []

    def log_message(self, *a): pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"data": [{"id": "bc-test"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.bodies.append(body)
        user = body["messages"][1]["content"]
        if "FAIL" in user:
            return self._send(500, {"error": "boom"})
        self._send(200, {"choices": [{"message": {"content": user.split("ANS=")[1]}}]})


class OfflineEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeVLLM)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_parse_reply_matches_vllmpolicy_rule(self):
        self.assertEqual(O.parse_reply("3", 5), ("3", True))
        self.assertEqual(O.parse_reply(" 2.\n", 5), ("2", True))
        self.assertEqual(O.parse_reply("Option 4", 5), ("4", True))
        self.assertEqual(O.parse_reply("7", 5), (None, False))      # out of range
        self.assertEqual(O.parse_reply("none", 5), (None, False))
        self.assertEqual(O.parse_reply(None, 5), (None, False))

    def test_request_body_matches_vllmpolicy(self):
        b = O.request_body("m", "sys", "usr")
        self.assertEqual((b["temperature"], b["max_tokens"]), (0, 3))
        self.assertEqual([m["role"] for m in b["messages"]], ["system", "user"])

    def make(self, d, answers):
        base = os.path.join(d, "x", "data"); os.makedirs(base)
        rows = []
        for i, (label, ans, mode, p_top) in enumerate(answers):
            r = row(f"s{i}", label, mode=mode, p_top=p_top)
            r["instruction"], r["input"] = "SYS", f"q{i} ANS={ans}" if ans != "FAIL" else f"q{i} FAIL"
            rows.append(r)
        with open(os.path.join(base, "test.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        json.dump({"params": {"confident": 0.7},
                   "baselines": {"global": {"test_acc": 0.5, "test_acc_confident": 0.5}}},
                  open(os.path.join(base, "dataset.json"), "w"))

    def run_eval(self, d, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return O.main(["--name", "x", "--root", d, "--model", "bc-test", "--tag", "t",
                           "--base", self.base, "--concurrency", "3", *extra])

    def test_end_to_end_metrics(self):
        FakeVLLM.bodies.clear()
        answers = [("1", "1", 1, 1.0),     # correct, ADD, confident
                   ("2", "3", 1, 1.0),     # wrong
                   ("0", "0", 0, 0.5),     # correct, REMOVE, not confident
                   ("1", "x", 0, 0.9),     # invalid reply
                   ("2", "FAIL", 1, 1.0)]  # server error
        with tempfile.TemporaryDirectory() as d:
            self.make(d, answers)
            s = self.run_eval(d)
            preds = [json.loads(l) for l in open(os.path.join(d, "x", "offline", "t", "predictions.jsonl"))]
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["accuracy"], 2 / 5)
        self.assertAlmostEqual(s["accuracy_confident"], 1 / 4)      # rows 0,1,3,4
        self.assertAlmostEqual(s["accuracy_add"], 1 / 3)
        self.assertAlmostEqual(s["accuracy_remove"], 1 / 2)
        self.assertAlmostEqual(s["invalid_rate"], 1 / 5)
        self.assertAlmostEqual(s["error_rate"], 1 / 5)
        self.assertEqual(s["confusion"]["1"], {"1": 1, "invalid": 1})
        self.assertEqual(s["baselines"]["global"]["test_acc"], 0.5)
        self.assertEqual([p["label"] for p in preds], ["1", "2", "0", "1", "2"])   # input order kept
        self.assertTrue(all(b["messages"][0]["content"] == "SYS" for b in FakeVLLM.bodies))

    def test_limit(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, [("1", "1", 1, 1.0)] * 4)
            self.assertEqual(self.run_eval(d, "--limit", "2")["n"], 2)

    def test_refuses_unserved_model_and_unreachable_server(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, [("1", "1", 1, 1.0)])
            with self.assertRaises(SystemExit) as e, contextlib.redirect_stdout(io.StringIO()):
                O.main(["--name", "x", "--root", d, "--model", "other", "--tag", "t", "--base", self.base])
            self.assertIn("not served", str(e.exception))
            with self.assertRaises(SystemExit) as e, contextlib.redirect_stdout(io.StringIO()):
                O.main(["--name", "x", "--root", d, "--model", "bc-test", "--tag", "t",
                        "--base", "http://127.0.0.1:9"])
            self.assertIn("not reachable", str(e.exception))


if __name__ == "__main__":
    unittest.main()
