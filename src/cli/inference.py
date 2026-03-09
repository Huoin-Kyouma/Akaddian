"""
Inference and MBR (Minimum Bayes Risk) decoding pipeline.
Supports multi-model ensembling with cross-model MBR.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import sacrebleu

PROJECT_DIR = Path(__file__).parent.parent
TASK_PREFIX = "translate Akkadian to English: "


def load_model(model_path: str, device: str = "cuda"):
    """Load a fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_candidates(
    model, 
    tokenizer, 
    source_text: str, 
    num_candidates: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    max_length: int = 512,
    device: str = "cuda",
) -> List[str]:
    """Generate multiple translation candidates via sampling."""
    
    input_text = TASK_PREFIX + source_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=1,  # Sampling, not beam search
        )
    
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [c.strip() for c in candidates]


def generate_candidates_beam(
    model,
    tokenizer,
    source_text: str,
    num_beams: int = 10,
    num_return: int = 10,
    max_length: int = 512,
    device: str = "cuda",
) -> List[str]:
    """Generate candidates via diverse beam search."""
    
    input_text = TASK_PREFIX + source_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return
        )
    
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [c.strip() for c in candidates]


def compute_chrf_matrix(candidates: List[str]) -> np.ndarray:
    """Compute pairwise chrF++ scores between all candidates."""
    n = len(candidates)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 100.0  # Perfect self-score
            else:
                score = sacrebleu.sentence_chrf(candidates[i], [candidates[j]], word_order=2)
                matrix[i, j] = score.score
    
    return matrix


def mbr_decode(candidates: List[str], utility_metric: str = "chrf++") -> str:
    """
    Select the best candidate using Minimum Bayes Risk decoding.
    
    The candidate with the highest average utility (chrF++) to all 
    other candidates is selected as the consensus translation.
    """
    if len(candidates) == 0:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    candidates = unique_candidates
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Compute pairwise scores
    matrix = compute_chrf_matrix(candidates)
    
    # MBR: select candidate with highest average score to all others
    avg_scores = matrix.mean(axis=1)
    best_idx = np.argmax(avg_scores)
    
    return candidates[best_idx]


def mbr_decode_batch(
    all_candidates: List[List[str]], 
    batch_size: int = 32,
) -> List[str]:
    """MBR decode a batch of candidate lists."""
    results = []
    for i, candidates in enumerate(all_candidates):
        if i % 100 == 0:
            print(f"  MBR decoding {i}/{len(all_candidates)}...")
        result = mbr_decode(candidates)
        results.append(result)
    return results


def cross_model_ensemble(
    models_and_tokenizers: List[tuple],
    source_texts: List[str],
    candidates_per_model: int = 30,
    temperature: float = 0.8,
    max_length: int = 512,
    device: str = "cuda",
) -> List[str]:
    """
    Cross-model MBR ensemble: pool candidates from multiple models,
    then apply MBR decoding on the combined candidate pool.
    """
    all_results = []
    
    for text_idx, source in enumerate(source_texts):
        if text_idx % 50 == 0:
            print(f"Processing {text_idx}/{len(source_texts)}...")
        
        # Collect candidates from all models
        all_candidates = []
        
        for model, tokenizer in models_and_tokenizers:
            # Sampling candidates
            sampling_candidates = generate_candidates(
                model, tokenizer, source,
                num_candidates=candidates_per_model,
                temperature=temperature,
                max_length=max_length,
                device=device,
            )
            all_candidates.extend(sampling_candidates)
            
            # Also add beam search candidates for diversity
            beam_candidates = generate_candidates_beam(
                model, tokenizer, source,
                num_beams=5,
                num_return=5,
                max_length=max_length, 
                device=device,
            )
            all_candidates.extend(beam_candidates)
        
        # MBR on pooled candidates
        best = mbr_decode(all_candidates)
        all_results.append(best)
    
    return all_results


def post_process_translation(
    translation: str,
    source: str,
    ne_dict: Optional[dict] = None,
) -> str:
    """Post-process a translation output."""
    import re
    
    if not translation:
        return ""
    
    # 1. Fix capitalization (sentence start)
    if translation and translation[0].islower():
        translation = translation[0].upper() + translation[1:]
    
    # 2. Handle gap markers — if source has gaps, don't hallucinate content
    source_gaps = source.count("<gap>") + source.count("<big_gap>")
    if source_gaps > 0:
        # Ensure translation has appropriate gap markers
        # Common convention: [...] for gaps
        pass  # Model should learn this from training data
    
    # 3. Clean up common artifacts
    translation = re.sub(r'\s+', ' ', translation).strip()
    translation = re.sub(r'\s+([.,;:!?])', r'\1', translation)
    
    # 4. Strip trailing whitespace and fix quotes
    translation = translation.strip()
    
    return translation


def run_inference(args):
    """Run full inference pipeline."""
    print("=" * 60)
    print("Deep Past Challenge - Inference Pipeline")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data
    test_df = pd.read_csv(PROJECT_DIR / "data" / "test.csv")
    source_texts = test_df["transliteration"].tolist()
    print(f"Loaded {len(source_texts)} test examples")
    
    # Load NE dictionary
    ne_dict = None
    ne_path = PROJECT_DIR / "processed" / "named_entities.json"
    if ne_path.exists():
        with open(ne_path) as f:
            ne_dict = json.load(f)
        print(f"Loaded {len(ne_dict)} named entities")
    
    # Load models
    model_paths = args.model_paths.split(",")
    print(f"Loading {len(model_paths)} models...")
    
    models_and_tokenizers = []
    for path in model_paths:
        path = path.strip()
        print(f"  Loading {path}...")
        model, tokenizer = load_model(path, device)
        models_and_tokenizers.append((model, tokenizer))
    
    # Generate translations
    if len(models_and_tokenizers) > 1:
        print(f"\nRunning cross-model MBR ensemble with {args.candidates_per_model} candidates per model...")
        translations = cross_model_ensemble(
            models_and_tokenizers=models_and_tokenizers,
            source_texts=source_texts,
            candidates_per_model=args.candidates_per_model,
            temperature=args.temperature,
            max_length=args.max_length,
            device=device,
        )
    else:
        print(f"\nRunning single-model inference with MBR decoding ({args.total_candidates} candidates)...")
        model, tokenizer = models_and_tokenizers[0]
        
        all_candidates = []
        for i, source in enumerate(source_texts):
            if i % 50 == 0:
                print(f"  Generating candidates for {i}/{len(source_texts)}...")
            
            candidates = generate_candidates(
                model, tokenizer, source,
                num_candidates=args.total_candidates,
                temperature=args.temperature,
                max_length=args.max_length,
                device=device,
            )
            all_candidates.append(candidates)
        
        print("Running MBR decoding...")
        translations = mbr_decode_batch(all_candidates)
    
    # Post-process
    print("Post-processing translations...")
    translations = [
        post_process_translation(t, s, ne_dict) 
        for t, s in zip(translations, source_texts)
    ]
    
    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": translations,
    })
    
    output_path = PROJECT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"\nSample outputs:")
    for i in range(min(3, len(submission))):
        print(f"  [{i}] {translations[i][:100]}...")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for Akkadian→English")
    
    parser.add_argument("--model_paths", type=str, required=True,
                        help="Comma-separated paths to fine-tuned models")
    parser.add_argument("--candidates_per_model", type=int, default=30,
                        help="Candidates per model for cross-model MBR")
    parser.add_argument("--total_candidates", type=int, default=50,
                        help="Total candidates for single-model MBR")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max generation length")
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
