"""Integration tests: the real tools/counselctl.py against a stub repo and a fake docker (tests/fakes/).
No Docker, torch or root needed; uses 127.0.0.1 ports 8000-8003 (skipped if they are busy).
    python3 -m unittest discover -s tests -v          # unit + integration, about a minute
"""
import json, os, signal, socket, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stubrepo  # noqa: E402

CTL = os.path.join(HERE, "..", "tools", "counselctl.py")
FAKES = os.path.join(HERE, "fakes")


def ports_busy():
    busy = []
    for p in range(8000, 8004):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) == 0:
                busy.append(p)
    return busy


SPEC = """
name = "{name}"
workers = 2
[eval]
seeds = "{seeds}"
baselines = {baselines}
{extra_eval}
{watch}
{models}
"""
OLD = '[[models]]\nlabel = "old"\nfrom = "a3c/oldrun"\n'
NEW = '[[models]]\nlabel = "new"\ntrain = {{ code = "fix6", episodes = {eps} }}\n'


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (b := ports_busy()):
            raise unittest.SkipTest(f"ports {b} are in use (stop any load servers first)")
        for f in ("docker", "uvicorn"):
            os.chmod(os.path.join(FAKES, f), 0o755)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = stubrepo.build(self.tmp.name)
        self.env = dict(os.environ, PATH=FAKES + os.pathsep + os.environ["PATH"],
                        FAKE_DOCKER_STATE=os.path.join(self.tmp.name, "dockerstate"),
                        COUNSELCTL_CONTAINER_ROOT=os.path.join(self.tmp.name, "croot"))
        for k in [k for k in self.env if k.startswith(("STUB_", "FAKE_NO", "FAKE_LEAVE"))]:
            del self.env[k]

    def tearDown(self):
        subprocess.run(["pkill", "-f", os.path.join(FAKES, "uvicorn")], capture_output=True)
        self.tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def spec(self, name="t1", seeds="1000-1002", models=OLD, extra_eval="", watch="",
             baselines='["noop", "random", "heur", "const:1x2"]'):
        p = os.path.join(self.tmp.name, f"{name}-{abs(hash((seeds, models, extra_eval, watch, baselines)))}.toml")
        with open(p, "w") as f:
            f.write(SPEC.format(name=name, seeds=seeds, models=models, extra_eval=extra_eval, watch=watch,
                                baselines=baselines))
        return p

    def run_ctl(self, spec, *flags, timeout=120, **env):
        e = dict(self.env, **env)
        r = subprocess.run([sys.executable, CTL, "--repo", self.repo, "run", spec, "--poll", "0.2", *flags],
                           env=e, capture_output=True, text=True, timeout=timeout)
        r.all = r.stdout + r.stderr
        return r

    def start_ctl(self, spec, **env):
        e = dict(self.env, **env)
        log = open(os.path.join(self.tmp.name, "bg.log"), "w")
        self.addCleanup(log.close)
        return subprocess.Popen([sys.executable, CTL, "--repo", self.repo, "run", spec, "--poll", "0.2"],
                                env=e, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

    def out(self, name="t1"):
        return os.path.join(self.repo, "results", "eval", name)

    def rows(self, name="t1"):
        with open(os.path.join(self.out(name), "episodes.jsonl")) as f:
            return [json.loads(l) for l in f if l.strip()]

    def report(self, name="t1"):
        with open(os.path.join(self.out(name), "report.md")) as f:
            return f.read()

    def wait_for(self, cond, timeout=30):
        t = time.time()
        while time.time() - t < timeout:
            if cond():
                return True
            time.sleep(0.1)
        return False

    # ---------------------------------------------------------------- happy path
    def test_train_eval_report_then_resume(self):
        sp = self.spec(models=OLD + NEW.format(eps=6))
        r = self.run_ctl(sp)
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("training t1-new", r.all)
        rows = self.rows()
        # 4 baselines + old (3 ckpts) + new (model2/4/6) x 2 modes = 16 arms x 3 seeds
        self.assertEqual(len(rows), 48)
        self.assertEqual({x["slo_used"] for x in rows}, {150})                   # spec conditions reached the env
        self.assertEqual({x["freq_used"] for x in rows}, {6666})
        rep = self.report()
        self.assertIn("old_m100_g | 3 | 0 |", rep)
        self.assertIn("= const_1x2 on every seed", rep)                          # const checkpoint detected
        self.assertIn("trained in this run", rep)
        with open(os.path.join(self.out(), "provenance.json")) as f:
            prov = json.load(f)
        self.assertIn("old/model100.pt", prov["checkpoint_sha256"])
        with open(os.path.join(self.repo, "results", "RUNS.md")) as f:
            self.assertIn("| t1 |", f.read())
        # same file again: training reused, nothing re-run
        rep1 = self.report()
        r2 = self.run_ctl(sp)
        self.assertEqual(r2.returncode, 0, r2.all)
        results = lambda txt: txt[txt.index("## Results"):txt.index("## Provenance")]
        self.assertEqual(results(self.report()), results(rep1))                # identical results on a re-run
        self.assertIn("training reused", r2.all)
        self.assertIn("0 episodes to run, 48 already done", r2.all)
        self.assertEqual(len(self.rows()), 48)

    def test_more_seeds_only_runs_new_ones(self):
        self.assertEqual(self.run_ctl(self.spec(seeds="1000-1001")).returncode, 0)
        r = self.run_ctl(self.spec(seeds="1000-1003"))
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("20 episodes to run, 20 already done", r.all)
        self.assertEqual(len(self.rows()), 40)

    def test_no_train_excludes_untrained_model(self):
        r = self.run_ctl(self.spec(models=OLD + NEW.format(eps=4)), "--no-train")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertFalse(any(x["arm"].startswith("new_") for x in self.rows()))
        self.assertIn("| new | trained here", self.report())
        self.assertIn("not trained yet (--no-train); excluded", self.report())

    def test_check_runs_nothing(self):
        r = self.run_ctl(self.spec(), "--check")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("OK: would train 0 model(s)", r.all)
        self.assertFalse(os.path.exists(self.out()))

    # ---------------------------------------------------------------- refusals before anything runs
    def test_refusals(self):
        self.assertEqual(self.run_ctl(self.spec()).returncode, 0)
        cases = [
            (self.spec(extra_eval='modes = ["greedy"]'), 2, "already exists with different settings"),
            (self.spec(name="b1", baselines='["random", "heur", "const:3x7"]'), 1, "no such shape"),
            (self.spec(name="b2", models=OLD.replace("oldrun", "nothing")), 1, "no modelN.pt"),
            (self.spec(name="b3", models=OLD + "checkpoints = [100, 900]\n"), 1, "checkpoints [900] not in"),
            (self.spec(name="b4", models=NEW.format(eps=2).replace("fix6", "fix99")), 1, "is not a commit"),
        ]
        for sp, code, msg in cases:
            with self.subTest(msg=msg):
                r = self.run_ctl(sp)
                self.assertEqual(r.returncode, code, r.all)
                self.assertIn(msg, r.all)
        bad = os.path.join(self.tmp.name, "broken.toml")
        with open(bad, "w") as f:
            f.write('name = "x"\nname = "y"\n')
        r = self.run_ctl(bad)
        self.assertEqual(r.returncode, 2); self.assertIn("is not valid TOML", r.all)
        self.assertNotIn("Traceback", r.all)
        os.makedirs(self.out("byhand"))
        r = self.run_ctl(self.spec(name="byhand"))
        self.assertEqual(r.returncode, 2); self.assertIn("not made by counselctl", r.all)
        r = self.run_ctl(self.spec(name="b5"), FAKE_NO_IMAGE="1")
        self.assertEqual(r.returncode, 1); self.assertIn("No such image", r.all)

    def test_conditions_mismatch_warned(self):
        r = self.run_ctl(self.spec(models='[[models]]\nlabel = "oth"\nfrom = "a3c/otherslo"\n'))
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("slo trained 250 vs eval 150", self.report())

    def test_uncommitted_eval_code_is_not_used(self):
        with open(os.path.join(self.repo, "tools", "eval_harness.py"), "a") as f:
            f.write("\nraise SystemExit('uncommitted edit was used')\n")
        r = self.run_ctl(self.spec())
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("uncommitted changes in tools/model/load-gen are NOT used", r.all)

    # ---------------------------------------------------------------- failures and recovery
    def test_failed_episodes_reported_then_retried(self):
        r = self.run_ctl(self.spec(), STUB_FAIL_SEED="1001")
        self.assertEqual(r.returncode, 1, r.all)
        self.assertIn("| old_m200_g | 2 | 1 |", self.report())                 # 2 ok, 1 failed
        r = self.run_ctl(self.spec())
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("6 episodes to run", r.all)                                # only the 6 failed ones
        self.assertIn("Episodes: 30 done, 0 failed", self.report())

    def test_broken_checkpoint_gets_its_own_section(self):
        r = self.run_ctl(self.spec(models='[[models]]\nlabel = "brk"\nfrom = "a3c/broken"\n'))
        self.assertEqual(r.returncode, 1)
        rep = self.report()
        self.assertIn("## Arms with no successful episode", rep)
        self.assertIn("brk_m500_g | 3 | `RuntimeError('checkpoint could not be loaded')`", rep)

    def test_driver_refuses_changed_harness_and_shows_log(self):
        r = self.run_ctl(self.spec(), STUB_BADSIG="1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("signature changed", r.all)                                 # from the log tail
        self.assertIn("Re-run the same command to resume", self.report())

    def test_stall_stops_and_resume_finishes(self):
        r = self.run_ctl(self.spec(watch="[watch]\nstall_min = 0.05"), STUB_HANG="1", timeout=60)
        self.assertEqual(r.returncode, 1, r.all)
        self.assertIn("stalled", r.all)
        r = self.run_ctl(self.spec(watch="[watch]\nstall_min = 0.05"))
        self.assertEqual(r.returncode, 0, r.all)
        self.assertEqual(len(self.rows()), 30)

    def test_leftover_container_is_cleared(self):
        os.makedirs(self.env["FAKE_DOCKER_STATE"], exist_ok=True)
        with open(os.path.join(self.env["FAKE_DOCKER_STATE"], "counsel-eval-t1.pid"), "w") as f:
            f.write("0")
        r = self.run_ctl(self.spec())
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("removing leftover container counsel-eval-t1", r.all)

    def test_divergent_training_excluded_and_not_retried(self):
        sp = self.spec(models=OLD + NEW.format(eps=12))
        r = self.run_ctl(sp, STUB_DIVERGE="1")
        self.assertEqual(r.returncode, 1, r.all)                                 # a model was excluded
        self.assertIn("WATCHDOG STOP: divergence", r.all)
        self.assertFalse(any(x["arm"].startswith("new_") for x in self.rows()))
        self.assertIn("training failed: watchdog: divergence", self.report())
        r = self.run_ctl(sp)
        self.assertIn("training failed earlier", r.all)
        self.assertNotIn("training t1-new", r.all)

    def test_training_crash_is_loud_and_rest_still_evaluated(self):
        r = self.run_ctl(self.spec(models=OLD + NEW.format(eps=4)), STUB_TRAIN_CRASH="1")
        self.assertEqual(r.returncode, 1, r.all)
        self.assertIn("TRAINING FAILED for new", r.all)
        self.assertIn("No module named 'torch_geometric'", r.all)                # log tail shown
        self.assertTrue(any(x["arm"].startswith("old_") for x in self.rows()))   # the rest still ran
        self.assertIn("no checkpoints written", self.report())

    # ---------------------------------------------------------------- interrupts and concurrency
    def test_ctrl_c_during_eval_then_resume_without_duplicates(self):
        sp = self.spec(seeds="1000-1019")
        p = self.start_ctl(sp, STUB_EP_SLOW="0.05")
        ep = os.path.join(self.out(), "episodes.jsonl")
        self.assertTrue(self.wait_for(lambda: os.path.exists(ep) and os.path.getsize(ep) > 2000))
        busy = self.run_ctl(sp)                                                  # second run while first is going
        self.assertEqual(busy.returncode, 2); self.assertIn("already running", busy.all)
        p.send_signal(signal.SIGINT)
        self.assertEqual(p.wait(timeout=30), 130)
        self.assertFalse(os.path.exists(os.path.join(self.repo, "runs", "t1", "running.pid")))
        n_partial = len(self.rows())
        self.assertLess(n_partial, 200)
        r = self.run_ctl(sp)
        self.assertEqual(r.returncode, 0, r.all)
        rows = self.rows()
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({(x["arm"], x["seed"]) for x in rows}), 200)

    def test_ctrl_c_during_training_is_retrained_next_time(self):
        sp = self.spec(models=OLD + NEW.format(eps=40))
        p = self.start_ctl(sp, STUB_TRAIN_SLOW="0.1")
        log = os.path.join(self.repo, "runs", "t1-new", "train.log")
        def ep3():
            if not os.path.exists(log):
                return False
            with open(log) as f:
                return "Ep: 3 " in f.read()
        self.assertTrue(self.wait_for(ep3))
        p.send_signal(signal.SIGINT)
        self.assertEqual(p.wait(timeout=30), 130)
        r = self.run_ctl(sp, STUB_TRAIN_SLOW="0.01")
        self.assertEqual(r.returncode, 0, r.all)
        self.assertIn("stopped with Ctrl-C last time: retraining", r.all)
        self.assertTrue(any(x["arm"] == "new_m40_g" for x in self.rows()))


if __name__ == "__main__":
    unittest.main()
