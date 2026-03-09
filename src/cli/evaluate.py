"""
Evaluation utilities for the Deep Past Challenge.
Computes the competition metric: geometric mean of BLEU and chrF++.
"""
import pandas as pd
import numpy as np
import sacrebleu
from typing import List, Tuple
from pathlib import Path


def compute_competition_metric(
    predictions: List[str], 
    references: List[str]
) -> dict:
    """
    Compute the competition metric: geometric mean of BLEU and chrF++.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        Dictionary with bleu, chrf++, and geo_mean scores
    """
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    
    geo_mean = np.sqrt(bleu.score * chrf.score) if (bleu.score > 0 and chrf.score > 0) else 0.0
    
    return {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "geo_mean": geo_mean,
        "bleu_detail": str(bleu),
        "chrf_detail": str(chrf),
    }


def evaluate_predictions(
    pred_file: str,
    ref_file: str,
    pred_col: str = "translation",
    ref_col: str = "translation",
):
    """Evaluate predictions from a file against references."""
    pred_df = pd.read_csv(pred_file)
    ref_df = pd.read_csv(ref_file)
    
    predictions = pred_df[pred_col].tolist()
    references = ref_df[ref_col].tolist()
    
    metrics = compute_competition_metric(predictions, references)
    
    print(f"Competition Metric (Geometric Mean): {metrics['geo_mean']:.2f}")
    print(f"  BLEU:   {metrics['bleu']:.2f}")
    print(f"  chrF++: {metrics['chrf++']:.2f}")
    
    return metrics


def cross_validate(
    df: pd.DataFrame, 
    train_fn, 
    predict_fn,
    n_folds: int = 5, 
    seed: int = 42,
) -> List[dict]:
    """
    Run k-fold cross-validation.
    
    Args:
        df: DataFrame with 'transliteration' and 'translation' columns
        train_fn: Function(train_df) -> model
        predict_fn: Function(model, test_df) -> List[str]
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        List of metric dictionaries per fold
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    fold_size = len(df) // n_folds
    
    all_metrics = []
    
    for fold in range(n_folds):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*40}")
        
        # Split
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(df)
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Train
        model = train_fn(train_df)
        
        # Predict
        predictions = predict_fn(model, val_df)
        
        # Evaluate
        references = val_df["translation"].tolist()
        metrics = compute_competition_metric(predictions, references)
        
        print(f"Fold {fold + 1} - Geo Mean: {metrics['geo_mean']:.2f}")
        all_metrics.append(metrics)
    
    # Summary
    avg_geo = np.mean([m["geo_mean"] for m in all_metrics])
    std_geo = np.std([m["geo_mean"] for m in all_metrics])
    print(f"\n{'='*40}")
    print(f"CV Results: Geo Mean = {avg_geo:.2f} ± {std_geo:.2f}")
    print(f"{'='*40}")
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_predictions(args.pred_file, args.ref_file)
