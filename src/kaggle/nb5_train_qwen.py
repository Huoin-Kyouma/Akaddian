# ==============================================================================
# NOTEBOOK 5: Train Qwen2.5-1.5B-Instruct with QLoRA (~4-6 hours on Kaggle T4x2)
# ==============================================================================
# Prerequisites: 
#   1. Attach "deep-past-initiative-machine-translation" dataset
#   2. Enable GPU (T4x2) & Internet
# After training: Download the ZIP from the Output tab for use in Inference
# ==============================================================================

# --- CELL 1: Dependencies ---
# !pip install -q transformers peft accelerate datasets sentencepiece

# --- CELL 2: Preprocessing ---
import os, re, shutil
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Prevent DataParallel OOM on T4x2

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
    for text_id in unique_ids:
        if text_id not in translit_map: continue
        text_sents = sentences[sentences["text_uuid"] == text_id].sort_values("sentence_obj_in_text")
        extracted = segment_document_to_sentences(translit_map[text_id], text_sents)
        if extracted:
            texts_with_sentences.add(text_id)
            for akk, eng in extracted:
                pairs.append({"transliteration": akk, "translation": eng, "text_id": text_id})

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

    train_df = train_df[["transliteration", "translation"]]
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df

train_df, val_df = prepare_data()

# --- CELL 3: Train Qwen2.5-1.5B with QLoRA ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RUN_NAME = "qwen2-5-1_5b-v1"

# Excellent Prompt Design for the LLM
SYSTEM_PROMPT = "You are an expert Assyriologist. Your task is to translate ancient Akkadian transliterations into fluent, grammatically correct English sentences."

# Pre-tokenize dataset instead of using trl formatters
print(f"Loading {MODEL_NAME} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nTranslate the following Akkadian text:\n{example['transliteration']}<|im_end|>\n<|im_start|>assistant\n{example['translation']}<|im_end|>"
    return tokenizer(text, truncation=True, max_length=384)

train_ds = HFDataset.from_pandas(train_df).map(tokenize_fn, remove_columns=train_df.columns.tolist())
val_ds = HFDataset.from_pandas(val_df).map(tokenize_fn, remove_columns=val_df.columns.tolist())

print(f"Loading {MODEL_NAME} model in 16-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={'': 0} # Force to GPU 0
)

# Prepare for LoRA
# model = prepare_model_for_kbit_training(model) # Removed: since we are not using k-bit anymore
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True  # Enables Weight-Decomposed Low-Rank Adaptation
)

# Crucial step: wrap the model with PEFT adapters since we are using standard Trainer
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    args=TrainingArguments(
        output_dir=f"{MODELS_DIR}/{RUN_NAME}",
        run_name=RUN_NAME,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
    ),
)

print("Training Qwen2.5 with QLoRA...")
trainer.train()

# Merge LoRA weights back into the base model so it's a standalone model for offline inference
print("Merging LoRA adapters into base model...")
best_model_path = f"{MODELS_DIR}/{RUN_NAME}/best_model"
trainer.save_model(best_model_path) # Saves adapter

from peft import AutoPeftModelForCausalLM
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    best_model_path,
    device_map="cpu",
    torch_dtype=torch.float16,
).merge_and_unload()

# Save final merged model
merged_output = f"{WORKING_DIR}/{RUN_NAME}_merged"
merged_model.save_pretrained(merged_output, safe_serialization=True)
tokenizer.save_pretrained(merged_output)

# Zip for download
shutil.make_archive(f"{WORKING_DIR}/{RUN_NAME}_payload", 'zip', merged_output)
print(f"Done! Download {RUN_NAME}_payload.zip from the Output tab.")
