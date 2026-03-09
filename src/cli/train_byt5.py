"""
Training script for ByT5 model on Akkadian → English translation.
Supports both ByT5-Small and ByT5-Base.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


TASK_PREFIX = "translate Akkadian to English: "


class AkkadianDataset(Dataset):
    """PyTorch dataset for Akkadian-English pairs."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_source_len=512, max_target_len=512):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source = TASK_PREFIX + str(row["transliteration"])
        target = str(row["translation"])
        
        # Tokenize
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def prepare_hf_dataset(df: pd.DataFrame, tokenizer, max_source_len=512, max_target_len=512):
    """Prepare Hugging Face Dataset for Seq2SeqTrainer."""
    
    def preprocess_function(examples):
        inputs = [TASK_PREFIX + str(t) for t in examples["transliteration"]]
        targets = [str(t) for t in examples["translation"]]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=max_source_len, 
            truncation=True, 
            padding=False,
        )
        
        labels = tokenizer(
            targets, 
            max_length=max_target_len, 
            truncation=True, 
            padding=False,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert to HF Dataset
    hf_dataset = HFDataset.from_pandas(df[["transliteration", "translation"]])
    
    # Tokenize
    tokenized = hf_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized


def compute_metrics_fn(tokenizer):
    """Return a compute_metrics function for the trainer."""
    import sacrebleu
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Strip whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute BLEU
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        
        # Compute chrF++
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
        
        # Geometric mean (competition metric)
        geo_mean = np.sqrt(bleu.score * chrf.score)
        
        return {
            "bleu": bleu.score,
            "chrf++": chrf.score,
            "geo_mean": geo_mean,
        }
    
    return compute_metrics


def train(args):
    """Run training."""
    print(f"=" * 60)
    print(f"Training {args.model_name}")
    print(f"=" * 60)
    
    # Load processed data
    processed_dir = PROJECT_DIR / "processed"
    train_df = pd.read_csv(processed_dir / "train_processed.csv")
    val_df = pd.read_csv(processed_dir / "val_processed.csv")
    
    # Optionally include lexicon pairs for curriculum learning phase 1
    if args.include_lexicon:
        lexicon_df = pd.read_csv(processed_dir / "lexicon_pairs.csv")
        # Only use eBL dictionary pairs (actual English definitions)
        lexicon_df = lexicon_df[lexicon_df["source"] == "ebl_dictionary"]
        # Rename columns to match
        lexicon_df = lexicon_df[["transliteration", "translation"]].copy()
        train_df = pd.concat([train_df[["transliteration", "translation"]], lexicon_df], 
                              ignore_index=True)
        print(f"Added {len(lexicon_df)} lexicon pairs to training data")
    
    # Filter valid pairs
    train_df = train_df.dropna(subset=["transliteration", "translation"])
    train_df = train_df[
        (train_df["transliteration"].str.len() > 2) &
        (train_df["translation"].str.len() > 2)
    ].reset_index(drop=True)
    
    val_df = val_df.dropna(subset=["transliteration", "translation"])
    val_df = val_df[
        (val_df["transliteration"].str.len() > 2) &
        (val_df["translation"].str.len() > 2)
    ].reset_index(drop=True)
    
    print(f"Training on {len(train_df)} pairs, validating on {len(val_df)} pairs")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Prepare datasets
    train_dataset = prepare_hf_dataset(train_df, tokenizer, args.max_source_len, args.max_target_len)
    val_dataset = prepare_hf_dataset(val_df, tokenizer, args.max_source_len, args.max_target_len)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    
    # Output directory
    output_dir = PROJECT_DIR / "models" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        
        # Training hyperparameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        
        # Mixed precision
        fp16=torch.cuda.is_available() and not args.bf16,
        bf16=args.bf16 and torch.cuda.is_available(),
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
        
        # Generation config for evaluation
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        
        # Other
        label_smoothing_factor=args.label_smoothing,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # Train
    print("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    
    # Save training metrics
    metrics = train_result.metrics
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    print(json.dumps(eval_results, indent=2))
    
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nModel saved to {output_dir / 'best_model'}")
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Train ByT5 for Akkadian→English translation")
    
    # Model
    parser.add_argument("--model_name", type=str, default="google/byt5-small",
                        choices=["google/byt5-small", "google/byt5-base"],
                        help="Model to fine-tune")
    parser.add_argument("--run_name", type=str, default="byt5-small-v1",
                        help="Name for this training run")
    
    # Data
    parser.add_argument("--max_source_len", type=int, default=512,
                        help="Max source sequence length (bytes)")
    parser.add_argument("--max_target_len", type=int, default=512,
                        help="Max target sequence length (bytes)")
    parser.add_argument("--include_lexicon", action="store_true",
                        help="Include lexicon pairs in training data")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluate every N steps")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 instead of FP16")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
