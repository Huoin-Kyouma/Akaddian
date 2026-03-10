# ==============================================================================
# NOTEBOOK 4 of 4: Cross-Model Ensemble Inference + Submission  (~2-4 hours)
# ==============================================================================
# Prerequisites:
#   1. Attach "deep-past-initiative-machine-translation" competition dataset
#   2. Upload model ZIPs from Notebooks 1-3 as Kaggle Datasets, then attach them
#      e.g. "deep-past-byt5-small-v1", "deep-past-byt5-base-v1", "deep-past-nllb-200-v1"
#   3. Enable GPU, Internet can be OFF (no downloads needed)
#
# IMPORTANT: Update the MODEL_CONFIGS paths below to match YOUR dataset slugs!
# ==============================================================================

# --- CELL 1: Dependencies ---
# !pip install -q transformers sentencepiece

# --- CELL 2: Configuration & Helpers ---
import os, re, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
WORKING_DIR = "/kaggle/working"

# ======================================================================
# UPDATE THESE PATHS to match the Kaggle Dataset slugs you uploaded!
# Each path should point to the folder containing config.json, model.safetensors, etc.
# ======================================================================
MODEL_CONFIGS = [
    {
        "name": "Qwen2.5-1.5B",
        "path": "/kaggle/input/deep-past-qwen2-5-1_5b-v1", # <-- UPDATE
        "type": "causal",
        "prefix": "",
        "n_cands": 15,  # Slightly fewer to fit in time block
    },
    {
        "name": "ByT5-Small",
        "path": "/kaggle/input/deep-past-byt5-small-v1",   # <-- UPDATE
        "type": "seq2seq",
        "prefix": "translate Akkadian to English: ",
        "n_cands": 5,   # Reduced to avoid 9-hour timeout
    },
    {
        "name": "ByT5-Base",
        "path": "/kaggle/input/deep-past-byt5-base-v1",    # <-- UPDATE
        "type": "seq2seq",
        "prefix": "translate Akkadian to English: ",
        "n_cands": 5,   # Reduced to avoid 9-hour timeout
    },
    {
        "name": "NLLB-200",
        "path": "/kaggle/input/deep-past-nllb-200-v1",     # <-- UPDATE
        "type": "seq2seq",
        "prefix": "",  # NLLB does NOT use a task prefix
        "n_cands": 15,
    },
    {
    "name": "Qwen2.5-1.5B-DoRA-v2",
    "path": "/kaggle/input/deep-past-qwen2-5-dora-v2",  # your second model's slug
    "type": "causal",
    "prefix": "",
    "n_cands": 10,  # reduce candidates if adding more models to stay within 9hrs
},

]

# --- Named Entity helpers ---
def normalize_entity(entity):
    entity = entity.replace("š", "sh").replace("ṣ", "s").replace("ṭ", "t")
    entity = entity.replace("á", "a").replace("é", "e").replace("í", "i").replace("ú", "u")
    entity = entity.replace("à", "a").replace("è", "e").replace("ì", "i").replace("ù", "u")
    entity = entity.replace("ā", "a").replace("ē", "e").replace("ī", "i").replace("ū", "u")
    parts = entity.split("-")
    if not parts: return entity
    return parts[0].capitalize() + "".join(p for p in parts[1:])

def build_ne_dict():
    ne_dict = {}
    pub = pd.read_csv(f"{DATA_DIR}/published_texts.csv")
    pattern = re.compile(r'(?:KIŠIB|DUMU)\s+([a-záàéèíìúùṣšḫṭ][a-záàéèíìúùṣšḫṭ\-]+)', re.IGNORECASE)
    for _, row in pub.iterrows():
        if isinstance(row.get("transliteration"), str):
            for name in pattern.findall(row["transliteration"]):
                clean = name.strip("-")
                if len(clean) > 2: ne_dict[clean] = ne_dict.get(clean, 0) + 1
    return ne_dict

def get_char_ngrams(text, n=3):
    chars = text.replace(" ", "")
    if len(chars) < n: return [chars] if chars else []
    return [chars[i:i+n] for i in range(len(chars) - n + 1)]

def n_gram_overlap(cand, ref, n=3):
    cand_ngrams = get_char_ngrams(cand, n)
    ref_ngrams = get_char_ngrams(ref, n)
    if not cand_ngrams or not ref_ngrams: return 0.0
    
    from collections import Counter
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)
    
    overlap = sum(min(cand_counts[g], ref_counts[g]) for g in cand_counts)
    precision = overlap / len(cand_ngrams)
    recall = overlap / len(ref_ngrams)
    
    if precision + recall == 0: return 0.0
    # F-score with beta=2 (similar to chrF, prioritizing recall)
    return (5 * precision * recall) / (4 * precision + recall)

def mbr_decode(candidates):
    if len(candidates) <= 1: return candidates[0] if candidates else ""
    seen = set()
    unique = [c for c in candidates if c not in seen and not seen.add(c)]
    if len(unique) == 1: return unique[0]
    n = len(unique)
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += n_gram_overlap(unique[i], unique[j], n=3)
    return unique[np.argmax(scores)]

def post_process(translation, source, ne_dict):
    if not translation or pd.isna(translation) or not str(translation).strip(): 
        return "<gap>"
    translation = str(translation)
    for entity in ne_dict.keys():
        if len(entity) > 3 and entity in source:
            norm = normalize_entity(entity)
            translation = re.sub(re.escape(norm), norm, translation, flags=re.IGNORECASE)
            no_hyphen = entity.replace("-", "")
            if no_hyphen.lower() in translation.lower():
                translation = re.sub(re.escape(no_hyphen), norm, translation, flags=re.IGNORECASE)
    if translation and translation[0].islower():
        translation = translation[0].upper() + translation[1:]
    if (source.count("<gap>") + source.count("<big_gap>")) > 0 and "[" not in translation:
        translation += " [...]"
    translation = re.sub(r'\s+', ' ', translation).strip()
    return re.sub(r'\s+([.,;:!?])', r'\1', translation).strip()

# --- CELL 3: Load Models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

ne_dict = build_ne_dict()
print(f"Named entities: {len(ne_dict)}")

models = []
for cfg in MODEL_CONFIGS:
    if os.path.exists(cfg["path"]):
        print(f"  ✓ Loading {cfg['name']} from {cfg['path']}...")
        tok = AutoTokenizer.from_pretrained(cfg["path"])
        if cfg.get("type", "seq2seq") == "causal":
            tok.padding_side = "left"
            if not tok.pad_token:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.float16).to(device)
        else:
            mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg["path"]).to(device)
        mdl.eval()
        models.append({"model": mdl, "tokenizer": tok, "prefix": cfg["prefix"], "n_cands": cfg["n_cands"], "name": cfg["name"], "type": cfg.get("type", "seq2seq")})
    else:
        print(f"  ✗ {cfg['name']} not found at {cfg['path']}, skipping")

print(f"\nLoaded {len(models)} model(s) for ensemble.")
if not models:
    raise RuntimeError("No models found! Check your dataset paths in MODEL_CONFIGS.")

# --- CELL 4: Generate Submission ---
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
source_texts = test_df["transliteration"].tolist()
print(f"Test examples: {len(source_texts)}")

translations = []
batch_size = 8  # Keep small since multiple models are in GPU memory

for i in range(0, len(source_texts), batch_size):
    batch_source = source_texts[i : i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(source_texts) + batch_size - 1) // batch_size
    if batch_num % 10 == 1:
        print(f"  Batch {batch_num}/{total_batches}...")

    batch_all_candidates = [[] for _ in range(len(batch_source))]

    for cfg in models:
        m, t, prefix, nc = cfg["model"], cfg["tokenizer"], cfg["prefix"], cfg["n_cands"]
        is_causal = cfg["type"] == "causal"

        # Prepare inputs
        if is_causal:
            sys_prompt = "You are an expert Assyriologist. Your task is to translate ancient Akkadian transliterations into fluent, grammatically correct English sentences."
            input_texts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nTranslate the following Akkadian text:\n{text}<|im_end|>\n<|im_start|>assistant\n" for text in batch_source]
        else:
            input_texts = [prefix + text for text in batch_source]

        inputs = t(input_texts, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Sampling candidates
        with torch.no_grad():
            outputs = m.generate(**inputs, max_new_tokens=512, num_return_sequences=nc,
                                 do_sample=True, temperature=0.8, top_k=50, top_p=0.95, num_beams=1)
        
        if is_causal:
            cands = [c.strip() for c in t.batch_decode(outputs[:, input_len:], skip_special_tokens=True)]
        else:
            cands = [c.strip() for c in t.batch_decode(outputs, skip_special_tokens=True)]
            
        for j in range(len(batch_source)):
            batch_all_candidates[j].extend(cands[j * nc : (j + 1) * nc])

        # Beam search diversity candidates
        with torch.no_grad():
            outputs = m.generate(**inputs, max_new_tokens=512, num_beams=5, num_return_sequences=5)
            
        if is_causal:
            cands = [c.strip() for c in t.batch_decode(outputs[:, input_len:], skip_special_tokens=True)]
        else:
            cands = [c.strip() for c in t.batch_decode(outputs, skip_special_tokens=True)]
            
        for j in range(len(batch_source)):
            batch_all_candidates[j].extend(cands[j * 5 : (j + 1) * 5])

    for j, candidates in enumerate(batch_all_candidates):
        best = mbr_decode(candidates)
        best = post_process(best, batch_source[j], ne_dict)
        translations.append(best)

submission = pd.DataFrame({"id": test_df["id"], "translation": translations})
submission.to_csv(f"{WORKING_DIR}/submission.csv", index=False)

print(f"\n{'='*60}")
print(f"DONE! submission.csv saved ({len(submission)} rows)")
print(f"{'='*60}")
print("\nSample outputs:")
for i in range(min(5, len(submission))):
    print(f"  [{i}] {translations[i][:120]}...")
print("\nNow click 'Submit to Competition' on the output submission.csv!")
