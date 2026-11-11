"""bc_offline_eval.py -- score a vLLM-served model on a BC test set against the teacher's labels.

Standard library only (urllib + threads): runs on the host in any Python, no venv needed.
Requests are built exactly like llm_policy.VLLMPolicy (system + user message, temperature 0,
max_tokens 3, first integer in the reply = option index), so offline numbers use the same
serving path as the online eval.

  # serve the trained model (in ~/vllm-env), then:
  python3 tools/bc_offline_eval.py --name _smoke_heur --model bc-_smoke_heur-l1b-smoke \
      --tag l1b-smoke
  # zero-shot base model on the same test set, for comparison:
  python3 tools/bc_offline_eval.py --name _smoke_heur --model unsloth/Llama-3.2-1B-Instruct \
      --tag zeroshot-l1b

Output (results/bc/<name>/offline/<tag>/):
  predictions.jsonl   one row per test decision: label, reply, prediction, valid, latency
  summary.json        accuracy (all / confident / per mode), invalid + error rates, latency,
                      confusion matrix, and the bc_dataset baselines side by side
"""
import argparse, collections, json, os, re, statistics, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------- one request (mirrors VLLMPolicy)
def request_body(model, system, user):
    return {"model": model, "temperature": 0, "max_tokens": 3,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}


def parse_reply(txt, n_options):
    """Same rule as VLLMPolicy.get_action: first integer; valid only if it indexes an option.
    Returns (prediction as str or None, valid)."""
    m = re.search(r"\d+", txt or "")
    k = int(m.group()) if m else -1
    return (str(k), True) if 0 <= k < n_options else (None, False)


def post_json(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def query(base, model, row, timeout):
    t0 = time.perf_counter()
    try:
        out = post_json(f"{base}/v1/chat/completions",
                        request_body(model, row["instruction"], row["input"]), timeout)
        txt, err = out["choices"][0]["message"]["content"], None
    except Exception as e:                       # network/server error: counted, never crashes the run
        txt, err = None, repr(e)
    lat = time.perf_counter() - t0
    pred, valid = parse_reply(txt, row["meta"]["n_options"]) if err is None else (None, False)
    return {"episode": row["meta"]["episode"], "k": row["meta"].get("k"),
            "mode": row["meta"]["mode"], "comp": row["meta"].get("comp"),
            "p_top": row["meta"]["p_top"], "label": row["output"], "reply": txt,
            "pred": pred, "valid": valid, "error": err, "correct": pred == row["output"],
            "lat_ms": 1e3 * lat}


# ---------------------------------------------------------------- pre-flight
def served_models(base, timeout=5.0):
    with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
        return [m["id"] for m in json.loads(r.read())["data"]]


# ---------------------------------------------------------------- metrics
def acc(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else None


def summarise(preds, confident):
    conf = [p for p in preds if p["p_top"] >= confident]
    lat = sorted(p["lat_ms"] for p in preds if p["error"] is None)
    confusion = collections.defaultdict(collections.Counter)
    for p in preds:
        confusion[p["label"]][p["pred"] if p["valid"] else "invalid"] += 1
    return {
        "n": len(preds),
        "accuracy": acc(preds),
        "accuracy_confident": acc(conf), "n_confident": len(conf),
        "accuracy_add": acc([p for p in preds if p["mode"] == 1]),
        "accuracy_remove": acc([p for p in preds if p["mode"] == 0]),
        "accuracy_valid_only": acc([p for p in preds if p["valid"]]),
        "invalid_rate": sum(not p["valid"] and p["error"] is None for p in preds) / len(preds),
        "error_rate": sum(p["error"] is not None for p in preds) / len(preds),
        "lat_ms_mean": statistics.fmean(lat) if lat else None,
        "lat_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
    }


def comparison_lines(summary, baselines):
    lines = ["                         test_acc   confident",
             f"  STUDENT                {summary['accuracy']:.3f}      "
             + ("-" if summary["accuracy_confident"] is None else f"{summary['accuracy_confident']:.3f}")]
    for fs, b in (baselines or {}).items():
        c = "-" if b.get("test_acc_confident") is None else f"{b['test_acc_confident']:.3f}"
        lines.append(f"  rule: {fs:17} {b['test_acc']:.3f}      {c}")
    if baselines:
        best_fs, best = max(((fs, b["test_acc"]) for fs, b in baselines.items()), key=lambda x: x[1])
        gap = summary["accuracy"] - best
        lines.append(f"  student - best rule ({best_fs}) = {gap:+.3f}")
    return lines


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="dataset name under results/bc/")
    p.add_argument("--model", required=True, help="served model name (vLLM --served-model-name)")
    p.add_argument("--tag", required=True, help="output folder name, e.g. l1b-smoke or zeroshot-l1b")
    p.add_argument("--root", default=os.path.join("results", "bc"))
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--base", default="http://localhost:9000")
    p.add_argument("--concurrency", type=int, default=8, help="parallel requests (vLLM batches them)")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--confident", type=float, default=None,
                   help="p_top threshold; default = the one bc_dataset used")
    p.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke)")
    a = p.parse_args(argv)

    base_dir = os.path.join(a.root, a.name)
    data_path = os.path.join(base_dir, "data", f"{a.split}.jsonl")
    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found; run bc_dataset.py --name {a.name} first")
    ds_path = os.path.join(base_dir, "data", "dataset.json")
    ds = json.load(open(ds_path)) if os.path.exists(ds_path) else {}
    confident = a.confident if a.confident is not None else ds.get("params", {}).get("confident", 0.7)

    try:
        models = served_models(a.base)
    except Exception as e:
        raise SystemExit(f"vLLM not reachable at {a.base} ({e!r}); start it first")
    if a.model not in models:
        raise SystemExit(f"model {a.model!r} is not served at {a.base}; served: {models}")

    rows = [json.loads(l) for l in open(data_path) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        preds = list(ex.map(lambda r: query(a.base, a.model, r, a.timeout), rows))
    wall = time.time() - t0

    s = summarise(preds, confident)
    out = os.path.join(base_dir, "offline", a.tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "predictions.jsonl"), "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in preds)
    summary = {"dataset": a.name, "split": a.split, "model": a.model, "tag": a.tag,
               "base": a.base, "confident_threshold": confident, "wall_s": wall, **s,
               "baselines": ds.get("baselines") if a.split == "test" else None,
               "dataset_raw_sha256": ds.get("raw_sha256"),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)

    print(f"{a.model} on {a.name}/{a.split}: n={s['n']}  wall {wall:.1f}s  "
          f"latency mean {s['lat_ms_mean'] or 0:.0f} ms p95 {s['lat_ms_p95'] or 0:.0f} ms")
    print(f"accuracy {s['accuracy']:.3f} (ADD {s['accuracy_add'] if s['accuracy_add'] is not None else '-'}, "
          f"REMOVE {s['accuracy_remove'] if s['accuracy_remove'] is not None else '-'})   "
          f"invalid {s['invalid_rate']:.1%}  errors {s['error_rate']:.1%}")
    if a.split == "test":
        print("\n".join(comparison_lines(s, ds.get("baselines"))))
    print("confusion (label -> predicted):")
    for lab, row in s["confusion"].items():
        print(f"  {lab}: {row}")
    return summary


if __name__ == "__main__":
    main()
