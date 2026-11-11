"""bc_train.py -- LoRA behavioural cloning of a teacher dataset into a small LLM (unsloth).

Run on the HOST in ~/unsloth-env (not in Docker, not in ~/vllm-env), with vLLM stopped:
    source ~/unsloth-env/bin/activate
    python tools/bc_train.py --name _smoke_heur --run l1b-smoke --max_steps 20

Input  results/bc/<name>/data/{train,val}.jsonl + dataset.json   (from bc_dataset.py)
Output results/bc/<name>/models/<run>/
         adapter/              LoRA weights + tokenizer
         merged/               16-bit merged model, what vLLM serves (unless --no_merge)
         train_manifest.json   base model, hyperparameters, dataset.json sha256, library
                               versions, log history, final val loss, wall time

Teacher-agnostic: it only sees (system, input, output) rows. Before training it prints one
fully tokenised example and the tokens that carry loss, so a broken chat template or response
mask is caught in seconds instead of after a training run.
"""
import argparse, dataclasses, hashlib, inspect, json, os, platform, time

# chat-template markers used to mask everything except the assistant's answer
PARTS = {"llama": ("<|start_header_id|>user<|end_header_id|>\n\n",
                   "<|start_header_id|>assistant<|end_header_id|>\n\n"),
         "qwen":  ("<|im_start|>user\n", "<|im_start|>assistant\n")}


# ---------------------------------------------------------------- data (no GPU libraries needed)
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_messages(r):
    return [{"role": "system", "content": r["instruction"]},
            {"role": "user", "content": r["input"]},
            {"role": "assistant", "content": r["output"]}]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs(base):
    data = os.path.join(base, "data")
    need = [os.path.join(data, f) for f in ("train.jsonl", "val.jsonl", "dataset.json")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {missing}; run bc_dataset.py first")
    return data


def render(tok, rows, max_len):
    """Chat-template each row; drop rows that would be truncated (truncation could cut the label)."""
    out, dropped = [], 0
    for r in rows:
        text = tok.apply_chat_template(to_messages(r), tokenize=False)
        if len(tok(text, add_special_tokens=False).input_ids) > max_len:
            dropped += 1
            continue
        out.append({"text": text})
    return out, dropped


# ---------------------------------------------------------------- trl version compatibility
def sft_config_kwargs(SFTConfig, max_len):
    """trl renamed max_seq_length -> max_length; pick whichever this version has."""
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    key = "max_length" if "max_length" in fields else "max_seq_length"
    return {key: max_len}


def trainer_tokenizer_kwarg(SFTTrainer, tok):
    """trl renamed tokenizer -> processing_class; unsloth may wrap the signature in **kwargs."""
    params = inspect.signature(SFTTrainer.__init__).parameters
    return {"processing_class": tok} if "processing_class" in params or "tokenizer" not in params \
        else {"tokenizer": tok}


def loss_tokens(tok, example):
    """Decode the tokens that carry loss in one processed training example."""
    ids, labels = example["input_ids"], example["labels"]
    kept = [i for i, l in zip(ids, labels) if l != -100]
    return tok.decode(kept), len(kept), len(ids)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="dataset name under results/bc/")
    p.add_argument("--run", required=True, help="model run name, e.g. l1b-r16-e2")
    p.add_argument("--root", default=os.path.join("results", "bc"))
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--family", choices=list(PARTS), default="llama")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max_steps", type=int, default=-1, help="> 0 overrides epochs (smoke runs)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16, help="LoRA rank (alpha = r)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--load_4bit", action="store_true", help="QLoRA, for 3B+ on 6 GB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_merge", action="store_true", help="skip the 16-bit merged export")
    p.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    a = p.parse_args()

    base = os.path.join(a.root, a.name)
    data = check_inputs(base)
    out = os.path.join(base, "models", a.run)
    if os.path.exists(os.path.join(out, "train_manifest.json")) and not a.force:
        raise SystemExit(f"{out} already has a finished run; use a new --run or --force")
    os.makedirs(out, exist_ok=True)

    # unsloth must be imported before trl/transformers so its patches apply
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    import torch, transformers, trl
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; activate ~/unsloth-env on the host")
    free, total = torch.cuda.mem_get_info()
    if free < 0.6 * total:
        raise SystemExit(f"only {free / 2**30:.1f} of {total / 2**30:.1f} GiB GPU memory free; "
                         "stop vLLM (or any other GPU process) first")

    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len,
                                                   dtype=None, load_in_4bit=a.load_4bit)
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=a.seed)

    train, d_tr = render(tok, read_jsonl(os.path.join(data, "train.jsonl")), a.max_len)
    val, d_va = render(tok, read_jsonl(os.path.join(data, "val.jsonl")), a.max_len)
    print(f"train={len(train)} val={len(val)} dropped_too_long={d_tr + d_va}")
    if not train or not val:
        raise SystemExit("empty train or val after rendering")

    bf16 = is_bfloat16_supported()
    cfg = SFTConfig(
        dataset_text_field="text", per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, max_steps=a.max_steps,
        learning_rate=a.lr, warmup_ratio=0.05, optim="adamw_8bit", weight_decay=0.01,
        bf16=bf16, fp16=not bf16, logging_steps=10, eval_strategy="epoch", save_strategy="no",
        seed=a.seed, output_dir=os.path.join(out, "trainer"), report_to="none",
        **sft_config_kwargs(SFTConfig, a.max_len))
    trainer = SFTTrainer(model=model, train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(val), args=cfg,
                         **trainer_tokenizer_kwarg(SFTTrainer, tok))
    instr, resp = PARTS[a.family]
    trainer = train_on_responses_only(trainer, instruction_part=instr, response_part=resp)

    # ---- sanity check BEFORE spending GPU time: only the answer should carry loss
    ex = trainer.train_dataset[0]
    answer, n_loss, n_tok = loss_tokens(tok, ex)
    bos = tok.bos_token_id
    n_bos = sum(1 for i in ex["input_ids"][:3] if i == bos) if bos is not None else 0
    print(f"example 0: {n_tok} tokens, {n_loss} carry loss -> {answer!r}; leading BOS x{n_bos}")
    expected = read_jsonl(os.path.join(data, "train.jsonl"))[0]["output"]
    if n_loss == 0 or expected not in answer or n_loss > 8:
        raise SystemExit(f"response mask looks wrong (loss on {answer!r}, expected the label "
                         f"{expected!r} plus end-of-turn); check --family / chat template")
    if n_bos > 1:
        print("WARNING: duplicated BOS token in training text; the served prompt will differ")

    stats = trainer.train()
    final_eval = trainer.evaluate()
    model.save_pretrained(os.path.join(out, "adapter"))
    tok.save_pretrained(os.path.join(out, "adapter"))
    if not a.no_merge:
        model.save_pretrained_merged(os.path.join(out, "merged"), tok, save_method="merged_16bit")

    import unsloth, peft
    manifest = {
        "dataset": a.name, "run": a.run, "base_model": a.model, "family": a.family,
        "hyperparams": {k: v for k, v in vars(a).items()
                        if k not in ("name", "run", "root", "model", "family", "force")},
        "dataset_json_sha256": sha256_file(os.path.join(data, "dataset.json")),
        "rows": {"train": len(train), "val": len(val), "dropped_too_long": d_tr + d_va},
        "mask_check": {"loss_text": answer, "loss_tokens": n_loss, "leading_bos": n_bos},
        "precision": "bf16" if bf16 else "fp16",
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "train_loss": stats.training_loss, "final_eval": final_eval,
        "log_history": trainer.state.log_history,
        "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__,
                     "unsloth": getattr(unsloth, "__version__", None)},
        "outputs": {"adapter": "adapter", "merged": None if a.no_merge else "merged"},
        "wall_s": time.time() - t0, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(manifest, open(os.path.join(out, "train_manifest.json"), "w"), indent=1, default=str)
    print(f"done in {manifest['wall_s']:.0f}s; train_loss={stats.training_loss:.4f} "
          f"eval_loss={final_eval.get('eval_loss')}; peak GPU {manifest['peak_gpu_mem_gib']:.2f} GiB")
    print(f"serve with: vllm serve {os.path.join(out, 'merged')} --served-model-name bc-{a.name}-{a.run}")


if __name__ == "__main__":
    main()
