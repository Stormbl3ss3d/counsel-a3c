#!/usr/bin/env bash
# Acceptance test with REAL Docker (laptop only: needs counsel:t113 and the a3c/ checkpoints).
# Runs tests/golden/golden.toml (27 episodes, about a minute) and checks every (arm, seed) against the frozen
# fix6-eval-v2 numbers. Leaves nothing behind. Exit 0 = pass.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
NAME=golden-check
cleanup() { rm -rf "results/eval/$NAME" "runs/$NAME"; git worktree prune; }
cleanup
[ -f tests/golden/expected.json ] || { echo "no tests/golden/expected.json: run 'python3 tests/golden/golden.py freeze results/eval/fix6-eval-v2' once"; exit 2; }
set +e
python3 tools/counselctl.py run tests/golden/golden.toml --no-runs-md --poll 10
rc=$?
set -e
[ $rc -eq 0 ] || { echo "counselctl run failed (exit $rc); output kept in results/eval/$NAME and runs/$NAME"; exit 1; }
set +e
python3 tests/golden/golden.py check "results/eval/$NAME"
g=$?
set -e
[ $g -eq 0 ] && cleanup || echo "output kept in results/eval/$NAME for inspection"
exit $g
