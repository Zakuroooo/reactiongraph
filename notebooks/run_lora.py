"""Part 5 on a Kaggle GPU: LoRA fine-tune vs base, few-shot and retrieval prompting.

Task: given reactants and reagents, predict the product. RDKit grades it, so no human labels.
All four methods answer the same held-out test set. Whole patents were held out, so nothing leaks.
Outputs land in OUT as each method finishes, so a crash still leaves the earlier numbers.

Kaggle: see notebooks/KAGGLE-STEPS.md. Local smoke test: REACTIONGRAPH_SMOKE=1 python notebooks/run_lora.py
"""
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import grade, mcnemar_exact, wilson  # noqa: E402  (a copy ships next to this script)

SMOKE = os.environ.get("REACTIONGRAPH_SMOKE") == "1"
MODEL = os.environ.get("REACTIONGRAPH_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA_PATH = Path(os.environ.get("REACTIONGRAPH_DATA", "/kaggle/input/reactiongraph/lora_data.json.gz"))
OUT = Path(os.environ.get("REACTIONGRAPH_OUT", "/kaggle/working/outputs"))
SEED = 42
TRAIN_MINUTES = float(os.environ.get("REACTIONGRAPH_TRAIN_MINUTES", 0.2 if SMOKE else 45))
MAX_EPOCHS = 2
BATCH_TRAIN = 2 if SMOKE else 8
BATCH_EVAL = 2 if SMOKE else 16
LR = 2e-4
MAX_NEW_TOKENS = 10 if SMOKE else 128
N_EXAMPLES = 3

RESULTS = {"settings": {"model": MODEL, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
                        "train_minutes_budget": TRAIN_MINUTES, "smoke": SMOKE}, "methods": {}}


def save():
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")


def load_data():
    with gzip.open(DATA_PATH, "rt") as f:
        data = json.load(f)
    if SMOKE:
        data["train"], data["val"], data["test"] = data["train"][:16], data["val"][:4], data["test"][:8]
    print("data", {k: len(v) for k, v in data.items()}, flush=True)
    return data


SYSTEM = ("You are a chemistry assistant. Predict the major product of the reaction. "
          "Reply with only the product as one SMILES string, with no explanation.")


def question(row):
    return f"Reactants: {row['reactants']}\nReagents: {row['reagents'] or 'none'}\nProduct SMILES:"


def messages(row, examples=()):
    """Chat for one test reaction. Few-shot and retrieval differ only in which examples come in here."""
    chat = [{"role": "system", "content": SYSTEM}]
    for example in examples:
        chat.append({"role": "user", "content": question(example)})
        chat.append({"role": "assistant", "content": example["product"]})
    chat.append({"role": "user", "content": question(row)})
    return chat


def build_model(tokenizer):
    # float16 is fast on a GPU but very slow on a CPU, so the local smoke test stays in float32.
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def generate(model, tokenizer, chats, batch_size=BATCH_EVAL):
    """Greedy replies for a list of chats, in the same order. Halves the batch if the GPU runs out."""
    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths batch together
    replies = [None] * len(texts)
    i = 0
    while i < len(order):
        chunk = order[i:i + batch_size]
        try:
            batch = tokenizer([texts[j] for j in chunk], return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                output = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print("out of GPU memory; batch size now", batch_size, flush=True)
            continue
        new_tokens = output[:, batch["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for j, reply in zip(chunk, decoded, strict=True):
            replies[j] = reply
        i += len(chunk)
        print(f"  {i}/{len(order)}", end="\r", flush=True)
    return replies


def run_method(name, model, tokenizer, test, chats):
    """Answer the test set, grade it with RDKit, save every prediction and the summary."""
    model.eval()
    start = time.time()
    replies = generate(model, tokenizer, chats)
    seconds = time.time() - start
    graded = []
    with (OUT / f"predictions_{name}.jsonl").open("w") as f:
        for row, reply in zip(test, replies, strict=True):
            valid, exact = grade(reply, row["product"])
            graded.append((row["id"], valid, exact))
            f.write(json.dumps({"id": row["id"], "product": row["product"], "reply": reply,
                                "valid": valid, "exact": exact}) + "\n")
    n = len(graded)
    n_valid = sum(g[1] for g in graded)
    n_exact = sum(g[2] for g in graded)
    RESULTS["methods"][name] = {
        "n": n, "valid": n_valid, "exact": n_exact,
        "valid_pct": round(100 * n_valid / n, 2), "exact_pct": round(100 * n_exact / n, 2),
        "valid_ci95": [round(100 * x, 2) for x in wilson(n_valid, n)],
        "exact_ci95": [round(100 * x, 2) for x in wilson(n_exact, n)],
        "seconds_per_answer": round(seconds / n, 4),
    }
    save()
    print(name, RESULTS["methods"][name], flush=True)
    print("  example reply:", repr(replies[0][:120]), "| true product:", test[0]["product"], flush=True)
    return {g[0]: g[2] for g in graded}


def encode(tokenizer, row):
    """Token ids plus labels. Labels are -100 on the prompt, so loss is only on the product."""
    prompt = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["product"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def collate(tokenizer, rows):
    encoded = [encode(tokenizer, r) for r in rows]
    width = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in encoded]).to(DEVICE),
        "labels": torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in encoded]).to(DEVICE),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in encoded]).to(DEVICE),
    }


def mean_loss(model, tokenizer, rows):
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), BATCH_EVAL):
            total += float(model(**collate(tokenizer, rows[i:i + BATCH_EVAL])).loss)
            batches += 1
    return total / max(1, batches)


def train_lora(model, tokenizer, data):
    """Freeze the model, train small LoRA matrices beside attention, stop on the clock or on val loss."""
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()
    print("trainable params", trainable, "of", total, flush=True)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    rows = list(data["train"])
    random.Random(SEED).shuffle(rows)
    history = [{"step": 0, "val_loss": round(mean_loss(model, tokenizer, data["val"]), 4)}]
    print("val loss before training", history[0]["val_loss"], flush=True)
    deadline = time.time() + TRAIN_MINUTES * 60
    best = history[0]["val_loss"]
    step = 0
    stopped = "epochs_done"
    for epoch in range(MAX_EPOCHS):
        model.train()
        for i in range(0, len(rows), BATCH_TRAIN):
            batch = collate(tokenizer, rows[i:i + BATCH_TRAIN])
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = model(**batch).loss
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step} train loss {float(loss):.4f}", flush=True)
            if time.time() > deadline:
                stopped = "time_budget"
                break
        val_loss = round(mean_loss(model, tokenizer, data["val"]), 4)
        history.append({"step": step, "epoch": epoch, "val_loss": val_loss})
        print("epoch", epoch, "val loss", val_loss, flush=True)
        if val_loss > best:
            stopped = "val_loss_rose"
            break
        best = val_loss
        model.train()
        if stopped == "time_budget":
            break
    RESULTS["lora"] = {"model": MODEL, "r": 16, "alpha": 32, "lr": LR, "batch_size": BATCH_TRAIN,
                       "trainable_params": trainable, "total_params": total,
                       "trainable_pct": round(100 * trainable / total, 3),
                       "train_rows": len(rows), "steps": step, "stopped_because": stopped,
                       "val_loss_history": history}
    save()
    model.save_pretrained(str(OUT / "lora_adapter"))
    return model


def compare(exact_by_method):
    """Exact McNemar test for each pair: on the questions where they disagree, is the gap real?"""
    out = {}
    names = list(exact_by_method)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = set(exact_by_method[a]) & set(exact_by_method[b])
            only_a = sum(exact_by_method[a][i_] and not exact_by_method[b][i_] for i_ in ids)
            only_b = sum(exact_by_method[b][i_] and not exact_by_method[a][i_] for i_ in ids)
            out[f"{a}_vs_{b}"] = {"only_" + a: only_a, "only_" + b: only_b,
                                  "p_value_mcnemar": round(mcnemar_exact(only_a, only_b), 4)}
    return out


def main():
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", flush=True)
    if DEVICE != "cuda" and not SMOKE:
        raise SystemExit("No GPU found. In Kaggle open Settings > Accelerator > GPU T4 x2 and run again.")
    RESULTS["settings"]["device"] = DEVICE
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    data = load_data()
    test = data["test"]
    RESULTS["settings"]["test_n"] = len(test)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = build_model(tokenizer)

    exact = {}
    exact["base"] = run_method("base", model, tokenizer, test, [messages(r) for r in test])
    exact["few_shot"] = run_method("few_shot", model, tokenizer, test,
                                   [messages(r, data["few_shot"]) for r in test])
    exact["retrieval"] = run_method("retrieval", model, tokenizer, test,
                                    [messages(r, r["neighbours"][:N_EXAMPLES]) for r in test])
    model = train_lora(model, tokenizer, data)
    exact["lora"] = run_method("lora", model, tokenizer, test, [messages(r) for r in test])

    RESULTS["pairwise"] = compare(exact)
    save()
    print(json.dumps({k: v["exact_pct"] for k, v in RESULTS["methods"].items()}, indent=2), flush=True)
    print(json.dumps(RESULTS["pairwise"], indent=2), flush=True)
    print("saved to", OUT, flush=True)


if __name__ == "__main__":
    DEVICE = "cpu"
    main()
