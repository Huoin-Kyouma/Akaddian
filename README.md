# Deep Past Challenge - Akkadian translation

This repository contains the codebase and infrastructure for the Deep Past Challenge: Machine Translation of Ancient Akkadian. 

The goal of this project is to accurately translate ancient Akkadian transliterations into fluent, grammatically correct English using modern NLP architectures, specifically addressing the low-resource constraints and linguistic complexities of cuneiform translation.

## 🚀 Key Features

*   **Multi-Model Ensemble Architecture:** Combines the strengths of byte-level tokenization (ByT5), highly multilingual sequence-to-sequence models (NLLB-200), and state-of-the-art Causal Language Models (Qwen2.5-1.5B).
*   **Minimum Bayes Risk (MBR) Decoding:** Implements cross-model candidate generation and MBR decoding optimized for `chrF++` and BLEU metrics to select the statistically optimal translation from a massive pool of hypotheses.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes Weight-Decomposed Low-Rank Adaptation (DoRA) and 16-bit precision training to effectively fine-tune large Causal LMs (like Qwen) within Kaggle's strict 15GB T4 GPU constraints.
*   **Robust Preprocessing Pipeline:** Explicitly models `<gap>` and `<big_gap>` tokens for damaged tablets and injects custom `akk_Latn` language tokens into standard multilingual vocabularies.
*   **Dual-Environment Support:** Cleanly separates pure CLI tools for local cluster training from standalone, zero-dependency Kaggle Notebooks for seamless remote execution.

## 📂 Repository Structure

The codebase is organized into local Command Line scripts (`src/cli/`) and standalone Kaggle Notebook scripts (`src/kaggle/`).

```text
.
├── docs/                   # AI planning, walkthroughs, and task tracking
├── run.sh                  # Main orchestrator pipeline for local execution
├── src/
│   ├── cli/                # Local training and evaluation modules
│   │   ├── evaluate.py     # Local validation script (geo-mean metric)
│   │   ├── inference.py    # Multi-model MBR inference engine
│   │   ├── preprocess.py   # Dataset alignment and sanitization
│   │   ├── train_byt5.py   # ByT5 fine-tuning script
│   │   ├── train_nllb.py   # NLLB-200 fine-tuning (custom vocabulary injection)
│   │   └── train_qwen.py   # Qwen2.5-1.5B DoRA fine-tuning script
│   └── kaggle/             # Standalone notebooks (copy-pasteable)
│       ├── nb1_train_byt5_small.py
│       ├── nb2_train_byt5_base.py
│       ├── nb3_train_nllb.py
│       ├── nb4_ensemble_inference.py
│       └── nb5_train_qwen.py
└── .gitignore
```

## 🛠️ Usage (Local Environment)

The `run.sh` script is the primary entrypoint for local data processing and training.

1.  **Preparation**
    Ensure your virtual environment is activated and you have placed the raw competition CSVs into `data/`.
    ```bash
    ./run.sh preprocess
    ```

2.  **Training Models Locally**
    You can train any of the specific model variations directly. The scripts use `argparse` for standard hyperparameter tuning.
    ```bash
    ./run.sh train-small --batch_size 8 --epochs 20
    ./run.sh train-base  --batch_size 4 --epochs 15
    ./run.sh train-nllb  --batch_size 4 --epochs 10
    ./run.sh train-qwen  --batch_size 2 --epochs 3 --lora_r 16
    ```

3.  **Local Inference & Evaluation**
    Generate candidates using MBR decoding and evaluate against the sample submission.
    ```bash
    ./run.sh infer models/qwen2-5-1_5b-v1/best_model
    ./run.sh evaluate submission.csv data/sample_submission.csv
    ```

4.  **Kaggle Upload Automation**
    Once local training is complete, pack the model weights into a Kaggle Dataset for use in the remote inference notebook.
    ```bash
    ./run.sh kaggle-dataset models/qwen2-5-1_5b-v1/best_model
    ```

## ☁️ Usage (Kaggle Environment)

To run this pipeline entirely on Kaggle servers (e.g., using the free T4x2 environment):

1.  Open Kaggle and create a new Notebook.
2.  Attach the `deep-past-initiative-machine-translation` competition dataset.
3.  Copy the entire contents of a script from `src/kaggle/` (e.g., `nb5_train_qwen.py`) and paste it into a single cell.
4.  Run all. The script contains its own dependency installation step at the top and will output a `_payload.zip` of the trained model into the `/kaggle/working` directory when complete.

---
*Developed for the Deep Past Initiative Machine Translation Challenge.*
