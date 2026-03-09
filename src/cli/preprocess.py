"""
Data preprocessing pipeline for the Deep Past Challenge.
Handles sentence alignment, cleaning, and augmentation.

Key insight: Sentences_Oare provides `first_word_number` (1-indexed word position
in the full transliteration) and `translation` for each sentence. We use these
to segment document-level transliterations into sentence-level pairs.
"""
import pandas as pd
import numpy as np
import re
import os
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def clean_transliteration(text: str) -> str:
    """Clean and normalize Akkadian transliteration text.
    Preserves diacritics — DO NOT normalize to ASCII.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\{large break\}', '<big_gap>', text)
    text = re.sub(r'\{break\}', '<gap>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"').strip()
    return text


def clean_translation(text: str) -> str:
    """Clean English translation text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"').strip()
    text = text.replace('""', '"')
    return text


def segment_document_to_sentences(
    transliteration: str, 
    sentence_infos: pd.DataFrame,
) -> list:
    """
    Segment a document-level transliteration into sentence-level pairs
    using word positions from Sentences_Oare.
    
    Args:
        transliteration: Full document transliteration
        sentence_infos: DataFrame of sentences for this text, sorted by sentence_obj_in_text
    
    Returns:
        List of (akkadian_sentence, english_translation) tuples
    """
    words = transliteration.split()
    total_words = len(words)
    
    if total_words == 0:
        return []
    
    # Get word positions for each sentence boundary
    boundaries = []
    for _, row in sentence_infos.iterrows():
        fw_num = row.get("first_word_number", None)
        translation = row.get("translation", "")
        
        if pd.isna(fw_num) or not isinstance(translation, str) or len(translation.strip()) < 3:
            continue
        
        fw_num = int(fw_num)
        if 1 <= fw_num <= total_words:
            boundaries.append({
                "start_word": fw_num - 1,  # Convert to 0-indexed
                "translation": clean_translation(translation),
            })
    
    if not boundaries:
        return []
    
    # Sort by start word position
    boundaries.sort(key=lambda x: x["start_word"])
    
    # Extract sentence transliterations using consecutive boundaries
    pairs = []
    for i, b in enumerate(boundaries):
        start = b["start_word"]
        # End is the start of next sentence, or end of document
        end = boundaries[i + 1]["start_word"] if i + 1 < len(boundaries) else total_words
        
        sentence_words = words[start:end]
        akkadian = " ".join(sentence_words)
        
        if len(akkadian.strip()) > 0 and len(b["translation"].strip()) > 3:
            pairs.append((akkadian, b["translation"]))
    
    return pairs


def build_sentence_level_training_data() -> pd.DataFrame:
    """
    Build the main sentence-level training dataset by:
    1. Segmenting train.csv documents into sentences using Sentences_Oare
    2. Using full document-level pairs for texts without sentence alignment
    3. Using published_texts.csv + Sentences_Oare for extra sentence pairs
    """
    train = pd.read_csv(DATA_DIR / "train.csv")
    train["transliteration"] = train["transliteration"].apply(clean_transliteration)
    train["translation"] = train["translation"].apply(clean_translation)
    
    sentences = pd.read_csv(DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv")
    pub = pd.read_csv(DATA_DIR / "published_texts.csv")
    
    # Build a transliteration lookup: text_id -> full transliteration
    translit_map = {}
    for _, row in train.iterrows():
        translit_map[row["oare_id"]] = clean_transliteration(row["transliteration"])
    for _, row in pub.iterrows():
        if isinstance(row.get("transliteration"), str) and len(row["transliteration"]) > 0:
            translit_map[row["oare_id"]] = clean_transliteration(row["transliteration"])
    
    pairs = []
    texts_with_sentences = set()
    texts_segmented = 0
    sentences_extracted = 0
    
    # Strategy 1: Segment texts using Sentences_Oare
    for text_id in sentences["text_uuid"].unique():
        if text_id not in translit_map:
            continue
        
        full_translit = translit_map[text_id]
        text_sentences = sentences[sentences["text_uuid"] == text_id].sort_values("sentence_obj_in_text")
        
        extracted = segment_document_to_sentences(full_translit, text_sentences)
        
        if extracted:
            texts_with_sentences.add(text_id)
            texts_segmented += 1
            sentences_extracted += len(extracted)
            
            for akk, eng in extracted:
                pairs.append({
                    "source": "sentence_aligned",
                    "transliteration": akk,
                    "translation": eng,
                    "text_id": text_id,
                })
    
    print(f"  Segmented {texts_segmented} texts → {sentences_extracted} sentence pairs")
    
    # Strategy 2: Keep full document-level pairs for texts without sentence alignment
    for _, row in train.iterrows():
        if row["oare_id"] not in texts_with_sentences:
            pairs.append({
                "source": "train_document",
                "transliteration": row["transliteration"],
                "translation": row["translation"],
                "text_id": row["oare_id"],
            })
    
    doc_only = len([p for p in pairs if p["source"] == "train_document"])
    print(f"  Added {doc_only} document-level pairs (no sentence alignment)")
    
    df = pd.DataFrame(pairs)
    
    # Filter: remove very short or empty pairs
    df = df[
        (df["transliteration"].str.len() > 5) & 
        (df["translation"].str.len() > 5)
    ].reset_index(drop=True)
    
    print(f"  Total after filtering: {len(df)} pairs")
    print(f"    Sentence-aligned: {len(df[df['source'] == 'sentence_aligned'])}")
    print(f"    Document-level: {len(df[df['source'] == 'train_document'])}")
    
    # Show quality stats
    sent_pairs = df[df["source"] == "sentence_aligned"]
    doc_pairs = df[df["source"] == "train_document"]
    
    print(f"\n  Quality stats:")
    print(f"    Sentence pairs - avg src: {sent_pairs['transliteration'].str.len().mean():.0f}, avg tgt: {sent_pairs['translation'].str.len().mean():.0f}")
    if len(doc_pairs) > 0:
        print(f"    Document pairs - avg src: {doc_pairs['transliteration'].str.len().mean():.0f}, avg tgt: {doc_pairs['translation'].str.len().mean():.0f}")
    
    return df


def build_lexicon_pairs() -> pd.DataFrame:
    """Build word/phrase-level pairs from the eBL dictionary."""
    pairs = []
    
    ebl = pd.read_csv(DATA_DIR / "eBL_Dictionary.csv")
    for _, row in ebl.iterrows():
        if isinstance(row.get("definition"), str) and isinstance(row.get("word"), str):
            word = row["word"].strip()
            defn = row["definition"].strip().strip('"')
            if len(word) > 0 and len(defn) > 0:
                pairs.append({
                    "source": "ebl_dictionary",
                    "transliteration": word,
                    "translation": defn,
                })
    
    df = pd.DataFrame(pairs)
    print(f"  Built {len(df)} dictionary pairs")
    return df


def build_named_entity_dict() -> dict:
    """Build Named Entity dictionary from available data."""
    ne_dict = {}
    pub = pd.read_csv(DATA_DIR / "published_texts.csv")
    
    name_pattern = re.compile(
        r'(?:KIŠIB|DUMU)\s+([a-záàéèíìúùṣšḫṭ][a-záàéèíìúùṣšḫṭ\-]+)', 
        re.IGNORECASE
    )
    
    for _, row in pub.iterrows():
        if isinstance(row.get("transliteration"), str):
            for name in name_pattern.findall(row["transliteration"]):
                clean_name = name.strip("-")
                if len(clean_name) > 2:
                    ne_dict[clean_name] = ne_dict.get(clean_name, 0) + 1
    
    return ne_dict


def create_train_val_split(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42):
    """Create train/val split, keeping texts together."""
    np.random.seed(seed)
    
    # Group by text_id to avoid data leakage
    text_ids = df["text_id"].unique()
    np.random.shuffle(text_ids)
    
    val_size = max(1, int(len(text_ids) * val_ratio))
    val_text_ids = set(text_ids[:val_size])
    
    val_df = df[df["text_id"].isin(val_text_ids)].reset_index(drop=True)
    train_df = df[~df["text_id"].isin(val_text_ids)].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)} pairs from {len(train_df['text_id'].unique())} texts")
    print(f"  Val: {len(val_df)} pairs from {len(val_df['text_id'].unique())} texts")
    
    return train_df, val_df


def main():
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("Deep Past Challenge - Data Preprocessing Pipeline v2")
    print("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build sentence-level training data
    print("\n[1/4] Building sentence-level training data...")
    train_data = build_sentence_level_training_data()
    
    # Step 2: Build lexicon pairs
    print("\n[2/4] Building dictionary pairs...")
    lexicon_pairs = build_lexicon_pairs()
    
    # Step 3: Build NE dictionary
    print("\n[3/4] Building Named Entity dictionary...")
    ne_dict = build_named_entity_dict()
    print(f"  Found {len(ne_dict)} unique named entities")
    
    # Step 4: Split and save
    print("\n[4/4] Creating train/val split...")
    train_split, val_split = create_train_val_split(train_data)
    
    # Save
    train_split.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    val_split.to_csv(PROCESSED_DIR / "val_processed.csv", index=False)
    lexicon_pairs.to_csv(PROCESSED_DIR / "lexicon_pairs.csv", index=False)
    
    with open(PROCESSED_DIR / "named_entities.json", "w") as f:
        json.dump(ne_dict, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print(f"  Training pairs: {len(train_split)}")
    print(f"  Validation pairs: {len(val_split)}")
    print(f"  Dictionary pairs: {len(lexicon_pairs)}")
    print(f"  Named entities: {len(ne_dict)}")
    print(f"  Output: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
