#  -- run in the *unsloth* env, never while vLLM is serving
from unsloth import FastLanguageModel, is_bfloat16_supported   # import before trl/transformers
import argparse, json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

PARTS = {"llama": ("<|start_header_id|>user<|end_header_id|>\n\n",
                   "<|start_header_id|>assistant<|end_header_id|>\n\n"),
         "qwen":  ("<|im_start|>user\n", "<|im_start|>assistant\n")}
p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", default="bc_out")
p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct"); p.add_argument("--family", default="llama")
p.add_argument("--epochs", type=int, default=2); p.add_argument("--max_len", type=int, default=2048)
p.add_argument("--load_4bit", action="store_true")
a = p.parse_args()

model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len,
                                               dtype=None, load_in_4bit=a.load_4bit)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=0)

rows = [json.loads(l) for l in open(a.data)]
val_eps = set(sorted({r["meta"]["episode"] for r in rows})[::10])   # hold out whole episodes
train, val, dropped = [], [], 0
for r in rows:
    msgs = [{"role": "system", "content": r["instruction"]}, {"role": "user", "content": r["input"]},
            {"role": "assistant", "content": r["output"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    if len(tok(text).input_ids) > a.max_len: dropped += 1; continue
    (val if r["meta"]["episode"] in val_eps else train).append({"text": text})
print(f"train={len(train)} val={len(val)} dropped_too_long={dropped}")

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=Dataset.from_list(train),
    eval_dataset=Dataset.from_list(val),
    args=SFTConfig(dataset_text_field="text", max_seq_length=a.max_len,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, num_train_epochs=a.epochs,
        learning_rate=2e-4, warmup_steps=20, optim="adamw_8bit", weight_decay=0.01,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        logging_steps=20, eval_strategy="steps", eval_steps=200, seed=0,
        output_dir=a.out, report_to="none"))
trainer = train_on_responses_only(trainer, instruction_part=PARTS[a.family][0], response_part=PARTS[a.family][1])
trainer.train()
model.save_pretrained(f"{a.out}/adapter"); tok.save_pretrained(f"{a.out}/adapter")
model.save_pretrained_merged(f"{a.out}/merged", tok, save_method="merged_16bit")