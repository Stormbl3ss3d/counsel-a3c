# Run manifest (counsel-a3c-fresh)

One row per run. Report numbers come from this file plus `results/`.

Tags:
- **[measured]**: run on the Victus, number read from a log or output file
- **[single-ep]**: one sampled episode (seed 1000, SLO 150, budget [2500,3600], 452 steps), from `tools/bc_extract.py`
- **[inferred]**: worked out from file timestamps or naming, not recorded at run time
- **[?]**: unknown, to fill in

Run folders live in `a3c/<name>/` (git-ignored, also in `~/counsel-backup-2026-11-10.tgz`).
Training logs: `logs/<name>.log`. Before the final report, re-evaluate every checkpoint with one
tools commit and record it here: eval commit = [?]

## Training runs: November (fresh clone of counsel-a3c)

| Run | Code (tag) | Image (torch) | Train conditions | -nw / eps | Wall time | Outcome | Eval so far |
|---|---|---|---|---|---|---|---|
| repro-f5-c3 (A) | train-repro 5b2b75d | counsel:local (2.0.0) | upstream (SLO 250, freq 1e6/U(200,300) drawn once, budget from hyperparams) | 4 / 1000 | 1:04:53 [measured] | actor loss diverges: -7e4 (ep 100) to -8.3e9 (ep 1000) | every ckpt: always 8 vCPU/32 GB, 177 breaches, 412.1 [single-ep] |
| repro-std16 (B) | train-repro 5b2b75d | counsel:local (2.0.0) | upstream | 16 / 500 | 0:14:37 [measured] | same divergence | same as A [single-ep] |
| fix23-f5-c3 | train-fix23 5309eda | counsel:local (2.0.0) | upstream | 4 / 1000 | 1:08:29 [measured] | no divergence; deterministic by ep 76 (actor loss exactly 0.0) | m100 79-119 breaches; m200+ no-op, 445 [single-ep] |
| t113-fix4-f5-c3 | train-fix4 471a8c7 | counsel:t113 (1.13.1) | upstream | 4 / 1000 | 1:04:59 [measured] | drifts to no-op | under TRAIN conditions no-op is reward-optimal (842.8, 77 breaches) [single-ep] |
| t113-fix5-f5-c3 | train-fix5 c4874ec | counsel:t113 (1.13.1) | SLO 150, freq 6666, budget 2500/3600, overrun 0.2 (`train_conditions.json`) | 4 / 1000 | 1:04:21 [measured] | partly stochastic later | m100-m400 const 1 vCPU/2 GB, 146 / 608.2; m900 83 / 679.6 [single-ep]; m900 greedy 125 breaches (eval_harness smoke test, 2026-11-10) |

`repro-std16` was named `std16` in the fresh clone until 2026-11-10; renamed to match `bc_data`, `bc_logs` and `logs/repro-std16.log`.

## Training runs: October (counsel-a3c-master; code tags fetched into this repo)

| Run | Code (tag) | Image (torch) | Train conditions | -nw / eps | Wall time | Outcome | Eval so far |
|---|---|---|---|---|---|---|---|
| fix1-f25-c3 | oct-fix1 c00fe97 [inferred] | counsel:local (2.0.0) | upstream | 4 / 100 | [?] | no collapse at 100 eps | [?] |
| fix1full-f25-c3 | oct-fix1b 61ec340 [inferred] | counsel:local (2.0.0) | upstream | 4 / 1000 | [?] | drifted to random (p_med 0.91 to 0.15) | 72-93 breaches |
| fix2-f25-c3 | oct-fix2 be3aa25 [inferred] (ent 0.001) | counsel:local (2.0.0) | upstream | 4 / stopped after ckpt 400 (shutdown) | ~1:00 to last write [inferred from file times] | near-deterministic, unstable | [?] |
| fix3-f25-c3 | oct-ent 5f793b0 [inferred: launched 1 s after that commit]; --ent value [?] | counsel:local (2.0.0) | upstream (SLO 250, freq 4672) | 4 / 1000 | 1:28:15 [measured] | collapsed by ep 100, entropy ~1e-26 | const 1 vCPU/4 GB, 101 breaches, 693.7 [single-ep] |
| std16 / op16 / up16 | pre-fix upstream + local additions (~oct-baseline) [inferred] | [?] | knob std 0.05 / op 0.03 / up 0.07 [inferred from names] | 4 / 500 | std16 0:55:11 [measured]; op16, up16 [?] | [?] | [?] |
| std-f{5,10,25,50,100}-c3, std-f5-c{5,10,20} (8 configs) | pre-fix (~oct-baseline) [inferred] | [?] | upstream | 4 / 1000 each (`scripts/run_counsel_local.sh`) | 1196 min total [measured, `logs/parallel_training_time_log.txt`] | all static policies (constant shape or no-op, p_top 1.00) | `results/raw/eval/bc_data/a3c_audit_*` |

## PPO runs (September, authors' MPI PPO, `data/`)

20 runs, 14-16 Sep: `std`, `op`, `up` and their `01-03` repeats, plus the 8 configs `std-f*-c*`.
Checkpoints are in `data/<name>/<name>_s0/pyt_save/model.pt` and logs in `logs/<name>_<timestamp>/`.
These are the PPO baselines used by `experiments/inference/quality_eval.py`. Per-run details: [?]

## Reference arms (eval conditions)

- [single-ep]: no-op 445 breaches; always-largest 177; random f25-c3 50 / 84 / 65 (reward 612 / 683 / 676)
- eval_harness smoke test 2026-11-10, seed 1000, f5-c3: no-op 445 breaches, matching the bc_extract figure

## Inference experiments (October, run from counsel-a3c-master)

Scripts are now in `experiments/inference/` (commit 6cf3462; only output paths changed). Raw outputs are in
`results/raw/inference/`, summaries in `results/inference/`. These runs predate the seeded load server.

| Experiment | Script | Result file | Status / caveats |
|---|---|---|---|
| Throughput: sequential vs parallel, 8 configs | benchmark_throughput.py | results/inference/benchmark_throughput_results.csv | [measured] 1.45-2.45x on 4 workers. Single run, unseeded. Script last modified 6 min after the CSV was written, so the committed script may not be exactly what produced it |
| Sequential vs parallel timing, 20 episodes | inference_seq_a3c.py, inference_parallel_a3c.py | results/raw/inference/infer_logs_seq/time_tracker_std_true_baseline.csv, results/raw/inference/infer_logs_parallel/time_tracker_std.csv | [measured] single runs, unseeded, from different days: sequential 2026-10-09, parallel 2026-10-29. inference_seq_a3c.py was edited after its output was written |
| Batched inference timing | batched_inference_benchmark.py | results/inference/batched_inference_times.csv | [measured] episodes completed 21-27 vs target 20, so per-episode times are not over equal work |
| Seq vs async vs batched, 20 episodes | compare_all_three_20_ep.py | results/inference/episode_20_inference_comparison.csv | [measured] unequal step counts between methods (e.g. f5-c3: 8000 / 8063 / 11364), so speedups are not like for like |
| Seq vs async vs batched, 2k steps | compare_all_three_2k_steps.py | results/inference/master_inference_comparison.csv | [measured] single run, unseeded |
| Decision quality: PPO (Sep) vs A3C (Oct, pre-fix) | quality_eval.py | results/inference/quality_summary.csv | [measured] compares against the pre-fix, collapsed A3C models. Summary file dated 2026-10-15, script 2026-10-09 |
| Scaling with worker count | generate_scaling_demo.py | held back: scaling_results.csv / scaling_plot.png stay in the archive | [measured] but unreliable: one repeat, 2000 steps total, unseeded; 8 workers need ports 8004-8007, which no compose file defines. Not for report use until re-run |
| A3C training wall time, 8 configs | scripts/run_counsel_local.sh | logs/parallel_training_time_log.txt | [measured] 1196 min, -nw 4 |

## Training-code notes from fix6 (tag fix6)
- Loss column order in `Ep:` lines: runs before commit eb27ae7 print (actor, critic); runs from eb27ae7 on print (critic, actor).
- fix6 Ep lines end with `| Ent: <mean policy entropy of the last update, decision steps only>` (nan = no actor update).
- Episode counter can end up to (nworkers - 1) past -e: each worker checks `< epochs` and then runs a whole episode. The final checkpoint is saved once, at == epochs.
- --resume restores weights only; SharedAdam moments restart (train_conditions.json: resume_optimizer_state = "fresh"). Resuming into the same run folder is refused before anything is written.
- Smoke test of fix6 (2026-11-11, counsel:t113, f5-c3, eval conditions, --ent 0.001, -nw 4): 8 episodes, then resume to 16 under a new name. It runs, logs Ent, resumes and refuses a same-folder resume. Not a training result.

## Evaluations: multi-seed (`tools/eval_harness.py`)

Paired design: every arm runs the same seeds, one fresh env per episode, 4 load-server ports.
Compare arms seed by seed with `tools/paired.py`. Findings: `claude/eval_fix6_findings_2026-11-11.md` (Project).

| Eval | Code | Image | Conditions | Arms | Output | Result |
|---|---|---|---|---|---|---|
| fix6-eval | eval_harness.py @ 6211281 [inferred: last commit touching eval_harness.py / eval_arms.py, 2026-11-10 19:07; envlogs last written 2026-11-11 03:26; uncommitted edits at run time can't be ruled out] | [?] | eval: SLO 150, freq 6666, budget [2500,3600], overrun 0.2, f5-c3; seeds 1000-1009 | noop, random, largest, heur (grow 0.5), const:1x2 + greedy and sampled for m100-m1000 of t113-fix5-f5-c3, t113-fix6-f5-c3-ent0, t113-fix6-f5-c3-ent001 (65 arms, 650 episodes, 0 failed) | results/eval/fix6-eval/ | [measured] breaches mean ± std: heur 47.9 ± 16; best RL fix5 m900 sample 105 ± 22, ent001 m300 sample 106 ± 33; random 106 ± 29; const:1x2 148 ± 31; noop 387 ± 140. Paired: heur beats every arm by ~58 (9-10/10 seeds); RL vs random not distinguishable (CI ±20-35). fix6-ent0 = const:1x2 exactly at every checkpoint; ent001 greedy = const:1x2 except m100 (~no-op). overrun_steps 0 for every arm (budget never binds) |

Supersedes the [single-ep] numbers above for these checkpoints: fix5 m900 is 105 (sample) / 158 (greedy) breaches over 10 seeds, not 83.

## Evaluations: counselctl

Rows written by `tools/counselctl.py run`; the report has the full table.

| Eval | Eval code | Image | Conditions | Models | Report | Headline |
|---|---|---|---|---|---|---|
| smoke-01 | 1e1d5b1 | 2fa7ed7e051d | SLO 150, budget [2500, 3600], seeds 1000-1001 | fix5, tiny | results/eval/smoke-01/report.md | [measured] best ckpt tiny_m6_g 41.5; random 100.0; heur 70.5; 0 failed |
| fix6-eval-v2 | 83a5ce0 | 2fa7ed7e051d | SLO 150, budget [2500, 3600], seeds 1000-1009 | fix5, e0, e1 | results/eval/fix6-eval-v2/report.md | [measured] best ckpt fix5_m900_s 105.4; random 106.3; heur 47.9; 0 failed |
| slm-bcheur100 | 7388aaf | 2fa7ed7e051d | SLO 150, budget [2500, 3600], seeds 1000-1009 | fix5 | results/eval/slm-bcheur100/report.md | [measured] best ckpt slm_bcheur 40.7; random 106.3; heur 47.9; 0 failed |
| slm-zeroshot-l1b | 7388aaf | 2fa7ed7e051d | SLO 150, budget [2500, 3600], seeds 1000-1009 |  | results/eval/slm-zeroshot-l1b/report.md | [measured] best ckpt slm_zsl1b 163.2; random 106.3; heur 47.9; 0 failed |
