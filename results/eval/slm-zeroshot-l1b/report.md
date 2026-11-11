# slm-zeroshot-l1b

Generated 2026-11-11T22:13:27 by `tools/counselctl.py run`. All numbers below are measured, from `episodes.jsonl` in this folder.

## Setup

- Conditions: SLO 150, freq 6666, budget [2500, 3600], overrun 0.2, knob 0.05, flavors 5, chain 3
- Seeds: 1000-1009 (10); modes: greedy; workers: 4
- Eval code: `7388aaf` (fix6-18-g7388aaf); image `sha256:2fa7ed7e051d`
- Episodes: 50 done, 0 failed

## Models

| label | source | checkpoints | trained on | note |
|---|---|---|---|---|

## Results (sorted by mean breaches; lower is better)

| arm | n | failed | breaches | 95% CI | reward/step | applied | util | overrun steps | note |
|---|---|---|---|---|---|---|---|---|---|
| heur | 10 | 0 | 47.9 | ±11.5 | 1.543 | 32.8 | 0.591 | 0.0 | baseline |
| random | 10 | 0 | 106.3 | ±20.8 | 1.473 | 86.9 | 0.646 | 0.0 | baseline |
| const_1x2 | 10 | 0 | 147.7 | ±22.4 | 1.378 | 90.2 | 0.738 | 0.0 | baseline |
| slm_zsl1b | 10 | 0 | 163.2 | ±17.7 | 1.355 | 103.5 | 0.741 | 0.0 |  |
| noop | 10 | 0 | 387.3 | ±97.4 | 0.744 | 0.0 | 0.921 | 0.0 | baseline |

CI = 95% t-interval of the mean over seeds. `= X on every seed` means breaches, actions applied, and mean CPU/memory match baseline X exactly on every seed: in practice the same policy.

## SLM arms (served by vLLM)

| arm | served model | queries/episode | invalid | errors | decision ms (mean / p95) |
|---|---|---|---|---|---|
| slm_zsl1b | `unsloth/Llama-3.2-1B-Instruct:operator` | 103.5 | 0.0% | 0.0% | 58 / 71 |

Invalid = the reply was not a listed option number; errors = the request failed. Both fall back to action 0 (no change). vLLM at temperature 0 is not guaranteed bit-exact across runs or batch sizes, so SLM rows can differ slightly on a re-run.

## Paired comparison vs reference arms

Same seeds, so each difference is on the same load trace. diff = arm minus reference (negative = fewer breaches). `*` = 95% interval excludes 0. W/T/L = seeds where the arm had fewer / equal / more breaches. No correction for the number of comparisons.

| arm | vs random | vs heur |
|---|---|---|
| const_1x2 | +41.4 [+4.9, +77.9] * 3/0/7 | +99.8 [+69.3, +130.3] * 0/1/9 |
| slm_zsl1b | +56.9 [+29.8, +84.0] * 1/0/9 | +115.3 [+100.2, +130.4] * 0/0/10 |
| noop | +281.0 [+173.0, +389.0] * 0/0/10 | +339.4 [+234.1, +444.7] * 0/0/10 |

## Automatic notes

- Best checkpoint arm: slm_zsl1b, 163.2 breaches (best baseline: heur)
- No arm ever exceeded the budget (overrun steps 0 everywhere): budget [2500, 3600] does not bind under these conditions.

## Provenance

```
{
 "created": "2026-11-11T22:12:47",
 "host": "Jeeva-Victus",
 "eval_commit": "7388aafeb5116b039979cca42a43c3dfb33d0fe8",
 "tool_describe": "fix6-18-g7388aaf",
 "counselctl_sha256": "858df2d950b2d6c77dc8143334a783ec20beb73c93b20ca533eaa85d1f0ebd92",
 "image": "counsel:t113",
 "image_id": "sha256:2fa7ed7e051dfde5ec82d268a5cd806c960ae819013db4f150f5569035634dbc",
 "checkpoint_sha256": {},
 "vllm": {
  "base": "http://localhost:9000",
  "container_base": "http://host.docker.internal:9000",
  "served": [
   {
    "id": "unsloth/Llama-3.2-1B-Instruct",
    "object": "model",
    "created": 1794415367,
    "owned_by": "vllm",
    "root": "unsloth/Llama-3.2-1B-Instruct",
    "parent": null,
    "max_model_len": 2048,
    "permission": [
     {
      "id": "modelperm-8b5e4c922697ffd0",
      "object": "model_permission",
      "created": 1794415367,
      "allow_create_engine": false,
      "allow_sampling": true,
      "allow_logprobs": true,
      "allow_search_indices": false,
      "allow_view": true,
      "allow_fine_tuning": false,
      "organization": "*",
      "group": null,
      "is_blocking": false
     }
    ]
   }
  ]
 }
}
```
Full `pip freeze` of the image is in `provenance.json`.
