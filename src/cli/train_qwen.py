"""
Training script for Qwen2.5-1.5B model on Akkadian → English translation.
Uses QLoRA / DoRA for parameter-efficient fine-tuning.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

SYSTEM_PROMPT = "You are an expert Assyriologist. Your task is to translate ancient Akkadian transliterations into fluent, grammatically correct English sentences."

def prepare_hf_dataset(df: pd.DataFrame, tokenizer, max_length=384):
    """Prepare Hugging Face Dataset for Causal LM Trainer."""
    def tokenize_fn(example):
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nTranslate the following Akkadian text:\n{example['transliteration']}<|im_end|>\n<|im_start|>assistant\n{example['translation']}<|im_end|>"
        return tokenizer(text, truncation=True, max_length=max_length)
    
    hf_dataset = HFDataset.from_pandas(df[["transliteration", "translation"]])
    tokenized = hf_dataset.map(
        tokenize_fn,
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing",
    )
    return tokenized

def train(args):
    print(f"=" * 60)
    print(f"Training {args.model_name} with DoRA")
    print(f"=" * 60)
    
    # Load processed data
    processed_dir = PROJECT_DIR / "processed"
    train_df = pd.read_csv(processed_dir / "train_processed.csv")
    val_df = pd.read_csv(processed_dir / "val_processed.csv")
    
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
    print(f"Loading {args.model_name} model and tokenizer in 16-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={'': 0} if torch.cuda.is_available() else "cpu",
    )
    
    model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=not args.disable_dora  # Enables Weight-Decomposed Low-Rank Adaptation
    )
    
    # Crucial step: wrap the model with PEFT adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset = prepare_hf_dataset(train_df, tokenizer, args.max_length)
    val_dataset = prepare_hf_dataset(val_df, tokenizer, args.max_length)
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Output directory
    output_dir = PROJECT_DIR / "models" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        fp16=True,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=args.dataloader_workers,
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Merging LoRA adapters into base model...")
    best_model_path = str(output_dir / "best_model_adapter")
    trainer.save_model(best_model_path) # Saves adapter
    
    from peft import AutoPeftModelForCausalLM
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        best_model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
    ).merge_and_unload()
    
    merged_output = str(output_dir / "best_model")
    merged_model.save_pretrained(merged_output, safe_serialization=True)
    tokenizer.save_pretrained(merged_output)
    
    print(f"\nMerged standalone model saved to {merged_output}")
    return 

def main():
    parser = argparse.ArgumentParser(description="Train Qwen with DoRA for Akkadian→English translation")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model to fine-tune")
    parser.add_argument("--run_name", type=str, default="qwen2-5-1_5b-v1",
                        help="Name for this training run")
    
    parser.add_argument("--max_length", type=int, default=384,
                        help="Max sequence length")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--disable_dora", action="store_true",
                        help="Disable DoRA and use standard LoRA")
    
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluate every N steps")
    parser.add_argument("--dataloader_workers", type=int, default=2,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
