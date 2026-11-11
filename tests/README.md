# Tests for tools/counselctl.py

| Layer | Command | Needs | Time |
|---|---|---|---|
| Unit | `python3 -m unittest tests.test_unit -v` | nothing | < 1 s |
| Integration | `python3 -m unittest tests.test_integration -v` | ports 8000-8003 free (stop any load servers) | ~30 s |
| Both | `python3 -m unittest discover -s tests` | as above | ~30 s |
| Acceptance (real Docker) | `bash tests/acceptance.sh` | `counsel:t113`, the `a3c/` checkpoints, `tests/golden/expected.json` | ~1 min |

Run from the repo root.

- **Unit** (`test_unit.py`): spec validation, statistics (t-intervals, dedupe), watchdog rules on real fix6 log
  lines, report contents, RUNS.md editing, and that the in-container eval driver is self-contained.
- **Integration** (`test_integration.py`): the real `counselctl.py` against a stub repo (`stubrepo.py`) and a
  fake `docker` / `uvicorn` (`fakes/`). Covers train + eval + resume, extra seeds, refusals, failed and
  retried episodes, broken checkpoints, training crashes, divergence, stalls, leftover containers, Ctrl-C in
  training and in eval, and concurrent runs. The fake docker maps mounts to symlinks under a temp folder via
  `COUNSELCTL_CONTAINER_ROOT`, so no root is needed.
- **Acceptance** (`acceptance.sh`): runs `golden/golden.toml` (27 episodes) with real Docker and requires every
  (arm, seed) to match the frozen `fix6-eval-v2` numbers exactly. Freeze once with
  `python3 tests/golden/golden.py freeze results/eval/fix6-eval-v2`. Exit 2 = a checkpoint file changed, so
  the golden numbers no longer apply.

Run unit + integration after any change to `counselctl.py`; run the acceptance test after any change to the
eval code (`tools/eval_harness.py`, `tools/eval_arms.py`, `model/`) or the image.
