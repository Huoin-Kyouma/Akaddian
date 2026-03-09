# ==============================================================================
# NOTEBOOK 1 of 4: Train ByT5-Small  (~3-5 hours on Kaggle T4)
# ==============================================================================
# Prerequisites: Attach "deep-past-initiative-machine-translation" dataset, enable GPU + Internet
# After training: Download the ZIP from the Output tab for use in Notebook 4
# ==============================================================================

# --- CELL 1: Dependencies ---
# !pip install -q transformers datasets sacrebleu sentencepiece

# --- CELL 2: Preprocessing ---
import os, re, json, shutil
import numpy as np
import pandas as pd
import torch
import sacrebleu
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)

DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
WORKING_DIR = "/kaggle/working"
MODELS_DIR = f"{WORKING_DIR}/models"
os.makedirs(MODELS_DIR, exist_ok=True)

def clean_transliteration(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\{large break\}', '<big_gap>', text)
    text = re.sub(r'\{break\}', '<gap>', text)
    return re.sub(r'\s+', ' ', text).strip().strip('"').strip()

def clean_translation(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip().strip('"').replace('""', '"')

def segment_document_to_sentences(transliteration, sentence_infos):
    words = transliteration.split()
    total_words = len(words)
    if total_words == 0: return []
    boundaries = []
    for _, row in sentence_infos.iterrows():
        fw_num, translation = row.get("first_word_number", None), row.get("translation", "")
        if pd.isna(fw_num) or not isinstance(translation, str) or len(translation.strip()) < 3: continue
        fw_num = int(fw_num)
        if 1 <= fw_num <= total_words:
            boundaries.append({"start_word": fw_num - 1, "translation": clean_translation(translation)})
    if not boundaries: return []
    boundaries.sort(key=lambda x: x["start_word"])
    pairs = []
    for i, b in enumerate(boundaries):
        start = b["start_word"]
        end = boundaries[i + 1]["start_word"] if i + 1 < len(boundaries) else total_words
        akkadian = " ".join(words[start:end])
        if len(akkadian.strip()) > 0 and len(b["translation"].strip()) > 3:
            pairs.append((akkadian, b["translation"]))
    return pairs

def prepare_data():
    print("Preprocessing...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    train["transliteration"] = train["transliteration"].apply(clean_transliteration)
    train["translation"] = train["translation"].apply(clean_translation)
    sentences = pd.read_csv(f"{DATA_DIR}/Sentences_Oare_FirstWord_LinNum.csv")
    pub = pd.read_csv(f"{DATA_DIR}/published_texts.csv")

    translit_map = {}
    for _, row in train.iterrows(): translit_map[row["oare_id"]] = clean_transliteration(row["transliteration"])
    for _, row in pub.iterrows():
        if isinstance(row.get("transliteration"), str): translit_map[row["oare_id"]] = clean_transliteration(row["transliteration"])

    pairs, texts_with_sentences = [], set()
    unique_ids = sentences["text_uuid"].unique()
    print(f"  Processing {len(unique_ids)} text IDs for sentence alignment...")
    for idx, text_id in enumerate(unique_ids):
        if idx % 500 == 0: print(f"    {idx}/{len(unique_ids)}...")
        if text_id not in translit_map: continue
        text_sents = sentences[sentences["text_uuid"] == text_id].sort_values("sentence_obj_in_text")
        extracted = segment_document_to_sentences(translit_map[text_id], text_sents)
        if extracted:
            texts_with_sentences.add(text_id)
            for akk, eng in extracted:
                pairs.append({"transliteration": akk, "translation": eng, "text_id": text_id})
    print(f"  Sentence alignment done: {len(pairs)} pairs")

    for _, row in train.iterrows():
        if row["oare_id"] not in texts_with_sentences:
            pairs.append({"transliteration": row["transliteration"], "translation": row["translation"], "text_id": row["oare_id"]})

    df = pd.DataFrame(pairs)
    df = df[(df["transliteration"].str.len() > 5) & (df["translation"].str.len() > 5)].reset_index(drop=True)

    np.random.seed(42)
    text_ids = df["text_id"].unique()
    np.random.shuffle(text_ids)
    val_size = max(1, int(len(text_ids) * 0.1))
    val_text_ids = set(text_ids[:val_size])
    val_df = df[df["text_id"].isin(val_text_ids)].reset_index(drop=True)
    train_df = df[~df["text_id"].isin(val_text_ids)].reset_index(drop=True)

    # Lexicon augmentation
    ebl = pd.read_csv(f"{DATA_DIR}/eBL_Dictionary.csv")
    lex = []
    for _, row in ebl.iterrows():
        if isinstance(row.get("definition"), str) and isinstance(row.get("word"), str):
            w, d = row["word"].strip(), row["definition"].strip().strip('"')
            if w and d: lex.append({"transliteration": w, "translation": d})
    train_df = pd.concat([train_df[["transliteration", "translation"]], pd.DataFrame(lex)], ignore_index=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Lexicon: {len(lex)}")
    return train_df, val_df

train_df, val_df = prepare_data()

# --- CELL 3: Train ByT5-Small ---
TASK_PREFIX = "translate Akkadian to English: "
MODEL_NAME = "google/byt5-small"
RUN_NAME = "byt5-small-v1"

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dp = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        dl = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu = sacrebleu.corpus_bleu(dp, [dl])
        chrf = sacrebleu.corpus_chrf(dp, [dl], word_order=2)
        geo = np.sqrt(bleu.score * chrf.score) if bleu.score > 0 and chrf.score > 0 else 0.0
        return {"bleu": bleu.score, "chrf++": chrf.score, "geo_mean": geo}
    return compute_metrics

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def tokenize(df):
    def fn(ex):
        inputs = [TASK_PREFIX + str(t) for t in ex["transliteration"]]
        mi = tokenizer(inputs, max_length=256, truncation=True, padding=False)
        mi["labels"] = tokenizer([str(t) for t in ex["translation"]], max_length=256, truncation=True, padding=False)["input_ids"]
        return mi
    return HFDataset.from_pandas(df[["transliteration", "translation"]]).map(fn, batched=True, remove_columns=["transliteration", "translation"])

train_ds, val_ds = tokenize(train_df.dropna()), tokenize(val_df.dropna())

trainer = Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(
        output_dir=f"{MODELS_DIR}/{RUN_NAME}", run_name=RUN_NAME,
        num_train_epochs=10, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, learning_rate=3e-4, warmup_steps=500, weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500, save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none", dataloader_num_workers=0,
    ),
    train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Training ByT5-Small...")
trainer.train()
best_dir = f"{MODELS_DIR}/{RUN_NAME}/best_model"
trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)
shutil.make_archive(f"{WORKING_DIR}/{RUN_NAME}_payload", 'zip', best_dir)
print(f"Done! Download {RUN_NAME}_payload.zip from the Output tab.")
