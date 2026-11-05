# Running Part 5 on Kaggle — your 6 steps

You run this under **your own Kaggle login**. Nobody needs your password, and you never paste a key
anywhere. The whole thing is one cell; the GPU does the rest. About 60–75 minutes, mostly waiting.

**Hinglish:** Apne hi Kaggle account se chalana hai. Ek hi cell chalani hai, baaki GPU khud karega.

---

## Step 1 — upload the data and the script as a Kaggle Dataset

1. Open <https://www.kaggle.com/datasets> → **New Dataset**.
2. Drag in all three files from `~/Desktop/reactiongraph/kaggle_upload/`:
   - `lora_data.json.gz` (the train/val/test split, 1 MB)
   - `run_lora.py` (the whole experiment)
   - `grading.py` (the RDKit grader)
3. Title it exactly **reactiongraph**, then **Create**.

The dataset will sit at `/kaggle/input/reactiongraph/`. If Kaggle gives it a different slug, you only
have to change the one path in Step 4.

## Step 2 — new notebook with a GPU

1. Open <https://www.kaggle.com/code> → **New Notebook**.
2. Right panel → **Add Input** → **Datasets** → pick your `reactiongraph`.
3. Right panel → **Session options** → **Accelerator** → **GPU T4 x2**.
4. Leave **Internet** **on** — the script downloads the base model from Hugging Face.

## Step 3 — install RDKit and PEFT

Paste into the first cell and run it:

```python
!pip install -q rdkit peft
```

## Step 4 — run the experiment

Paste into a second cell and run it. This is the long one.

```python
import os, shutil, subprocess, sys
os.environ["REACTIONGRAPH_DATA"] = "/kaggle/input/reactiongraph/lora_data.json.gz"
os.environ["REACTIONGRAPH_OUT"] = "/kaggle/working/outputs"
shutil.copy("/kaggle/input/reactiongraph/grading.py", "/kaggle/working/grading.py")
shutil.copy("/kaggle/input/reactiongraph/run_lora.py", "/kaggle/working/run_lora.py")
subprocess.run([sys.executable, "/kaggle/working/run_lora.py"], check=True)
```

**What you will see, in order:** `base`, then `few_shot`, then `retrieval` numbers; then LoRA training
loss; then `lora` numbers; then the pairwise comparison. Each method's numbers print as soon as it
finishes, and are written to `/kaggle/working/outputs/results.json` immediately — so if the session
dies at 80%, the earlier numbers are still saved.

**Expect weak numbers.** A 1.5B general model is bad at chemistry. LoRA may not beat retrieval. Whatever
comes out is what goes on the resume.

## Step 5 — download the outputs

Right panel → **Output** → download these five files into
`~/Desktop/reactiongraph/data/kaggle_outputs/`:

- `results.json`
- `predictions_base.jsonl`
- `predictions_few_shot.jsonl`
- `predictions_retrieval.jsonl`
- `predictions_lora.jsonl`

## Step 6 — tell me, and I re-grade them here

```bash
cd ~/Desktop/reactiongraph && .venv/bin/python -m src.grade_lora data/kaggle_outputs
```

This reads **only the raw replies** and grades them again with RDKit on this machine, writing
`results/lora.json`. So the resume number does not depend on trusting the notebook's own summary.
If the two disagree, `disagreements_with_notebook` is not zero and we investigate before using it.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `No GPU found` | Settings → Accelerator → GPU T4 x2, then re-run the cell |
| `No such file /kaggle/input/reactiongraph/...` | Your dataset slug differs; fix the path in Step 4 to match the left file browser |
| Out of GPU memory | The script halves its batch size itself and keeps going; let it run |
| Session ran out of time | Whatever finished is already in `results.json`; download it and we report those methods only |
| It is 8 PM and it is not done | Stop. We ship the resume without the LoRA bullet. That was always the plan. |

## What this is, in plain words

- **Base**: ask the model cold.
- **Few-shot**: show it the same 3 solved reactions every time.
- **Retrieval**: show it the 3 *most similar* training reactions (from Part 3's fingerprint search).
- **LoRA**: freeze the model's own weights, train small extra matrices on 20,000 training reactions,
  then ask it cold again.

All four answer the **same 500 held-out reactions**. Whole patents were held out, so a near-copy of a
test reaction cannot sit in the training set. RDKit canonicalises both answers and compares, so a
different spelling of the right product still counts as correct.

**Hinglish:** Chaaron methods ko wahi 500 sawaal milte hain. LoRA mein bada model freeze rehta hai,
sirf chhoti matrices train hoti hain.
