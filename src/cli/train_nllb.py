"""
Training script for NLLB-200 model on Akkadian → English translation.
Adds a custom 'akk_Latn' token to NLLB's vocabulary.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import sacrebleu
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

AKKADIAN_TOKEN = "akk_Latn"

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

def prepare_hf_dataset(df: pd.DataFrame, tokenizer, max_source_len=256, max_target_len=512):
    def tokenize_nllb(ex):
        tokenizer.src_lang = AKKADIAN_TOKEN
        mi = tokenizer(ex["transliteration"], max_length=max_source_len, truncation=True, padding=False)
        tokenizer.src_lang = "eng_Latn"
        mi["labels"] = tokenizer(text_target=ex["translation"], max_length=max_target_len, truncation=True, padding=False)["input_ids"]
        tokenizer.src_lang = AKKADIAN_TOKEN
        return mi
    return HFDataset.from_pandas(df[["transliteration", "translation"]]).map(tokenize_nllb, batched=True, remove_columns=["transliteration", "translation"])

def train(args):
    print(f"=" * 60)
    print(f"Training {args.model_name}")
    print(f"=" * 60)
    
    # Load processed data
    processed_dir = PROJECT_DIR / "processed"
    train_df = pd.read_csv(processed_dir / "train_processed.csv")
    val_df = pd.read_csv(processed_dir / "val_processed.csv")
    
    if args.include_lexicon:
        lexicon_df = pd.read_csv(processed_dir / "lexicon_pairs.csv")
        lexicon_df = lexicon_df[lexicon_df["source"] == "ebl_dictionary"]
        lexicon_df = lexicon_df[["transliteration", "translation"]].copy()
        train_df = pd.concat([train_df[["transliteration", "translation"]], lexicon_df], ignore_index=True)
        print(f"Added {len(lexicon_df)} lexicon pairs to training data")
    
    train_df = train_df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)
    
    print(f"Training on {len(train_df)} pairs, validating on {len(val_df)} pairs")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Add custom Akkadian language token
    tokenizer.add_tokens([AKKADIAN_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.src_lang = AKKADIAN_TOKEN
    tokenizer.tgt_lang = "eng_Latn"
    
    train_dataset = prepare_hf_dataset(train_df, tokenizer, args.max_source_len, args.max_target_len)
    val_dataset = prepare_hf_dataset(val_df, tokenizer, args.max_source_len, args.max_target_len)
    
    output_dir = PROJECT_DIR / "models" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir), run_name=args.run_name,
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available() and not args.bf16, bf16=args.bf16 and torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="geo_mean", greater_is_better=True,
        predict_with_generate=True, generation_max_length=args.max_target_len, generation_num_beams=4,
        report_to="none", dataloader_num_workers=args.dataloader_workers, remove_unused_columns=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100),
        compute_metrics=make_compute_metrics(tokenizer), callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print("Starting training...")
    trainer.train()
    
    best_dir = str(output_dir / "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    tokenizer.save_vocabulary(best_dir)
    
    eval_results = trainer.evaluate()
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nModel saved to {best_dir}")
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Train NLLB-200 for Akkadian→English translation")
    
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M",
                        help="Model to fine-tune")
    parser.add_argument("--run_name", type=str, default="nllb-200-v1",
                        help="Name for this training run")
    
    parser.add_argument("--max_source_len", type=int, default=256,
                        help="Max source sequence length")
    parser.add_argument("--max_target_len", type=int, default=512,
                        help="Max target sequence length")
    parser.add_argument("--include_lexicon", action="store_true",
                        help="Include lexicon pairs in training data")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 instead of FP16")
    parser.add_argument("--dataloader_workers", type=int, default=2,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
