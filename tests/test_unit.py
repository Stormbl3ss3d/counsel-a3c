"""Unit tests for tools/counselctl.py: spec validation, statistics, watchdog, report, RUNS.md.
No Docker, no git repo needed (except where a temp one is built). Run from the repo root:
    python3 -m unittest discover -s tests -v
"""
import json, math, os, sys, tempfile, unittest, xml.dom.minidom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import counselctl as c  # noqa: E402
from counselctl import _read, _write, _append, _load, _dump  # noqa: E402


def spec(**over):
    """A valid user spec (TOML-shaped dict) with overrides."""
    u = {"name": "t1", "models": [{"label": "old", "from": "a3c/oldrun"}]}
    u.update(over)
    return u


def resolved(**over):
    return c.validate(c.merge(spec(**over)))


class SpecTests(unittest.TestCase):
    def test_defaults_filled(self):
        s = resolved()
        self.assertEqual(s["conditions"]["slo"], 150)
        self.assertEqual(s["eval"]["modes"], ["greedy", "sample"])
        self.assertEqual(s["models"][0]["checkpoints"], "all")

    def test_freq_zero_means_1e6_over_slo(self):
        self.assertEqual(c.freq_of(resolved()["conditions"]), 6666)
        self.assertEqual(c.freq_of(resolved(conditions={"slo": 250})["conditions"]), 4000)
        self.assertEqual(c.freq_of(resolved(conditions={"freq": 5000})["conditions"]), 5000)

    def test_unknown_keys_rejected_by_name(self):
        for bad, where in (({"epochs": 3}, "top-level"), ({"eval": {"seed": "1"}}, "[eval]")):
            with self.assertRaises(c.SpecError) as e:
                resolved(**bad)
            self.assertIn(where, str(e.exception))
        with self.assertRaises(c.SpecError) as e:
            c.merge(spec(models=[{"label": "x", "from": "a", "episodes": 3}]))
        self.assertIn("episodes", str(e.exception))
        with self.assertRaises(c.SpecError) as e:
            c.merge(spec(models=[{"label": "x", "train": {"epochs": 3}}]))
        self.assertIn("epochs", str(e.exception))

    def test_all_value_errors_listed_together(self):
        with self.assertRaises(c.SpecError) as e:
            resolved(name="bad name", workers=12, conditions={"flavors": 7, "budget": [1], "overrun": 3})
        msg = str(e.exception)
        for part in ("name:", "workers", "flavors", "budget", "overrun"):
            self.assertIn(part, msg)

    def test_model_rules(self):
        cases = [
            ([{"label": "Old", "from": "a"}], "label required"),
            ([{"label": "x", "from": "a", "train": {}}], "exactly one of"),
            ([{"label": "x"}], "exactly one of"),
            ([{"label": "x", "from": "a"}, {"label": "x", "from": "b"}], "duplicate labels"),
            ([{"label": "x", "from": "a", "checkpoints": []}], "checkpoints"),
            ([{"label": "x", "train": {"ent": 0.5}}], "train.ent"),
        ]
        for models, want in cases:
            with self.subTest(want=want), self.assertRaises(c.SpecError) as e:
                resolved(models=models)
            self.assertIn(want, str(e.exception))

    def test_reference_must_be_a_baseline(self):
        with self.assertRaises(c.SpecError):
            resolved(eval={"baselines": ["random"], "reference": ["heur"]})

    def test_baseline_names(self):
        resolved(eval={"baselines": ["noop", "random", "largest", "heur", "heur:0.8", "const:1x2"],
                       "reference": ["random"]})
        with self.assertRaises(c.SpecError):
            resolved(eval={"baselines": ["const:1by2"], "reference": []})

    def test_parse_seeds(self):
        self.assertEqual(c.parse_seeds("1000-1003"), [1000, 1001, 1002, 1003])
        self.assertEqual(c.parse_seeds("5,7"), [5, 7])
        self.assertEqual(c.parse_seeds(42), [42])


class DriverTests(unittest.TestCase):
    def test_eval_driver_is_self_contained(self):
        """The driver runs inside the container: it must compile and must not call counselctl's host helpers."""
        import ast
        tree = ast.parse(c.EVAL_DRIVER)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertFalse(names & {"_read", "_write", "_append", "_load", "_dump", "sh", "SpecError"})
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [type("x", (), {"name": n.module})])}
        self.assertLessEqual(imported, {"inspect", "json", "os", "sys", "time", "multiprocessing",
                                        "eval_harness", "eval_arms", "traceback", "urllib"})


class ConditionTests(unittest.TestCase):
    def test_mismatch_reported_per_field(self):
        s = resolved()["conditions"]
        self.assertIsNone(c.condition_mismatch({"slo": 150, "freq": 6666, "budget": [2500, 3600],
                                                "overrun": 0.2, "knob": 0.05, "nconf": 5, "ncomp": 3}, s))
        m = c.condition_mismatch({"slo": 250, "freq": 4000, "budget": [600, 800]}, s)
        for part in ("slo trained 250", "freq trained 4000", "budget trained [600, 800]"):
            self.assertIn(part, m)
        self.assertIn("unknown", c.condition_mismatch(None, s))


class WatchdogTests(unittest.TestCase):
    FIX6 = ("w00 Ep: 6 | Loss:  371.915473470299 0.024568423628807068  | Util:  0.668399519165578  | "
            "Reward:  1.28795123564753  | Ent:  0.6923559308052063")       # real line from smoke-fix6-01

    def cfg(self, **k):
        return {**c.TOP["watch"], "window": 3, **k}

    def line(self, ep, critic=1.0, actor=-0.01, ent=0.5):
        return c.parse_ep(f"w00 Ep: {ep} | Loss:  {critic} {actor} | Util: 0.5 | Reward: 1.4 | Ent: {ent}",
                          "critic,actor")

    def test_real_fix6_line_columns(self):
        r = c.parse_ep(self.FIX6, "critic,actor")
        self.assertAlmostEqual(r["critic"], 371.915473470299)
        self.assertAlmostEqual(r["actor"], 0.024568423628807068)
        self.assertIsNone(c.Watchdog(self.cfg(), "critic,actor").feed(r))

    def test_upstream_order_and_no_ent(self):
        r = c.parse_ep("w00 Ep: 100 | Loss:  -68000.0 3.2  | Util:  0.5  | Reward:  0.4", "actor,critic")
        self.assertEqual((r["actor"], r["critic"], r["ent"]), (-68000.0, 3.2, None))
        self.assertEqual(c.Watchdog(self.cfg(), "actor,critic").feed(r)[0], "stop")

    def test_big_critic_is_not_divergence(self):
        self.assertIsNone(c.Watchdog(self.cfg(), "critic,actor").feed(self.line(1, critic=5e4)))

    def test_nan_stops(self):
        self.assertEqual(c.Watchdog(self.cfg(), "critic,actor").feed(self.line(1, actor="nan"))[0], "stop")

    def test_unknown_order_never_stops_on_size(self):
        r = c.parse_ep("w00 Ep: 1 | Loss:  -68000.0 3.2  | Util:  0.5  | Reward:  0.4", "unknown")
        self.assertIsNone(c.Watchdog(self.cfg(), "unknown").feed(r))

    def test_collapse_warns_once_then_again_after_recovery(self):
        w = c.Watchdog(self.cfg(), "critic,actor")
        ents = [0.5, 1e-5, 1e-5, 1e-5, 1e-5, 0.2, 1e-5, 1e-5, 1e-5]
        fired = [i for i, e in enumerate(ents) if w.feed(self.line(i, ent=e))]
        self.assertEqual(fired, [3, 8])

    def test_collapse_can_stop(self):
        w = c.Watchdog(self.cfg(on_collapse="stop"), "critic,actor")
        out = [w.feed(self.line(i, ent=1e-6)) for i in range(3)]
        self.assertEqual(out[-1][0], "stop")

    def test_non_ep_lines_ignored(self):
        for l in ("SLO: 150, Freq: 6666, Knob: 0.05", "COUNSELCTL: 2 load servers up", ""):
            self.assertIsNone(c.parse_ep(l, "critic,actor"))


class StatsTests(unittest.TestCase):
    def test_t_values(self):
        self.assertAlmostEqual(c.t975(9), 2.26)
        self.assertAlmostEqual(c.t975(11), 2.23)    # between table entries: falls back to the smaller df (wider)
        self.assertTrue(math.isnan(c.t975(0)))

    def test_ci_matches_hand_calculation(self):
        xs = [10, 12, 14, 16, 18]                    # mean 14, sd sqrt(10)
        self.assertAlmostEqual(c.ci(xs), 2.78 * math.sqrt(10) / math.sqrt(5), places=6)
        self.assertTrue(math.isnan(c.ci([3])))

    def test_load_rows_dedupes_and_drops_retried_failures(self):
        rows = [dict(arm="a", seed=1, breaches=5), dict(arm="a", seed=1, breaches=99),        # duplicate
                dict(arm="b", seed=1, error="x"), dict(arm="b", seed=1, breaches=7),           # retried
                dict(arm="c", seed=1, error="y"), dict(arm="c", seed=1, error="y")]            # still failing
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "e.jsonl")
            _write(p, "\n".join(json.dumps(r) for r in rows) + "\n")
            ok, bad = c.load_rows(p)
        self.assertEqual([(r["arm"], r["breaches"]) for r in ok], [("a", 5), ("b", 7)])
        self.assertEqual([(r["arm"], r["seed"]) for r in bad], [("c", 1)])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        self.out = os.path.join(self.repo, "results", "eval", "t1")
        os.makedirs(self.out)
        mdir = os.path.join(self.repo, "a3c", "oldrun", "model")
        os.makedirs(mdir)
        self.s = resolved(eval={"seeds": "1-4", "modes": ["greedy"], "baselines": ["random", "heur", "const:1x2"],
                                "reference": ["random", "heur"]})
        self.plan = [(self.s["models"][0], mdir, [100, 200])]
        rows = []
        for seed in (1, 2, 3, 4):
            def r(arm, b, applied=60, cpu=85.0):
                return dict(arm=arm, seed=seed, breaches=b, applied=applied, cpu_mean=cpu, mem_mean=200.0,
                            reward_per_step=1.4, util_mean=0.7, overrun_steps=0)
            rows += [r("random", 100 + seed), r("heur", 40 + seed), r("const_1x2", 150 + seed, 90, 81.0 + seed),
                     r("old_m100_g", 150 + seed, 90, 81.0 + seed),            # identical to const_1x2
                     r("old_m200_g", 99 + seed)]
        rows += [dict(arm="old_m300_g", seed=1, error="RuntimeError('cannot load')")]      # arm that never worked
        _write(os.path.join(self.out, "episodes.jsonl"), "\n".join(map(json.dumps, rows)) + "\n")
        self.prov = {"eval_commit": "abc1234def", "tool_describe": "fix6-1-gabc1234", "image_id": "sha256:feed"}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return c.build_report(self.repo, self.s, self.out, self.plan, self.prov, {}, ["a warning"])

    def test_order_collapse_paired_and_failures(self):
        order, by, bad = self.build()
        self.assertEqual(order, ["heur", "old_m200_g", "random", "const_1x2", "old_m100_g"])
        rep = _read(os.path.join(self.out, "report.md"))
        self.assertIn("= const_1x2 on every seed", rep)                    # collapse detector
        self.assertIn("| old_m200_g | -1.0 [-1.0, -1.0] * 4/0/0", rep)     # paired vs random: always 1 fewer
        self.assertIn("## Arms with no successful episode", rep)
        self.assertIn("old_m300_g | 1 | `RuntimeError('cannot load')`", rep)
        self.assertIn("budget [2500, 3600] does not bind", rep)
        self.assertIn("- a warning", rep)
        self.assertEqual(len(bad), 1)

    def test_seeds_outside_spec_ignored(self):
        self.s["eval"]["seeds"] = "1-2"
        _, by, _ = self.build()
        self.assertEqual(sorted(by["random"]["breaches"]), [1, 2])

    def test_summary_csv_and_svg(self):
        self.build()
        lines = _read(os.path.join(self.out, "summary.csv")).splitlines()
        self.assertTrue(lines[0].startswith("arm,n,breaches_mean"))
        self.assertTrue(lines[1].startswith("heur,4,42.5"))
        xml.dom.minidom.parse(os.path.join(self.out, "breaches.svg"))

    def test_runs_md_section_created_once_and_row_updated_in_place(self):
        p = os.path.join(self.repo, "results", "RUNS.md")
        hand = "# Run manifest\n\n## Training runs\n\n| Run | Note |\n|---|---|\n| hand | by hand |\n"
        _write(p, hand)
        order, by, bad = self.build()
        c.append_runs_md(self.repo, self.s, self.out, self.prov, order, by, bad)
        c.append_runs_md(self.repo, self.s, self.out, self.prov, order, by, bad)      # re-run: no new row
        self.s2 = dict(self.s, name="t2")
        c.append_runs_md(self.repo, self.s2, self.out, self.prov, order, by, bad)
        txt = _read(p)
        self.assertTrue(txt.startswith(hand))                                       # hand-written part untouched
        self.assertEqual(txt.count("## Evaluations: counselctl"), 1)
        self.assertEqual(txt.count("| t1 |"), 1)
        self.assertLess(txt.index("| t1 |"), txt.index("| t2 |"))

    def test_runs_md_row_lands_in_its_section_even_with_sections_after(self):
        p = os.path.join(self.repo, "results", "RUNS.md")
        _write(p, "# M\n")
        order, by, bad = self.build()
        c.append_runs_md(self.repo, self.s, self.out, self.prov, order, by, bad)
        _append(p, "\n## Later section\n\n| x | y |\n|---|---|\n| 1 | 2 |\n")
        c.append_runs_md(self.repo, dict(self.s, name="t2"), self.out, self.prov, order, by, bad)
        txt = _read(p)
        self.assertLess(txt.index("| t2 |"), txt.index("## Later section"))


if __name__ == "__main__":
    unittest.main()
