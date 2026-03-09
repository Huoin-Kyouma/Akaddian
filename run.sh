#!/bin/bash
# Deep Past Challenge - Main Pipeline
# Usage:
#   ./run.sh preprocess     - Preprocess data
#   ./run.sh train-small    - Train ByT5-Small locally
#   ./run.sh train-base     - Train ByT5-Base locally
#   ./run.sh train-nllb     - Train NLLB-200 locally
#   ./run.sh train-qwen     - Train Qwen2.5-1.5B with DoRA locally
#   ./run.sh infer          - Run local inference with MBR
#   ./run.sh evaluate       - Evaluate predictions locally

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

case "$1" in
    preprocess)
        echo "Running data preprocessing..."
        python -m src.cli.preprocess
        ;;
    
    train-small)
        echo "Training ByT5-Small locally..."
        python -m src.cli.train_byt5 \
            --model_name google/byt5-small \
            --run_name byt5-small-v1 \
            --epochs 20 \
            --batch_size 8 \
            --gradient_accumulation 4 \
            --learning_rate 3e-4 \
            --warmup_steps 500 \
            --include_lexicon \
            "${@:2}"
        ;;
    
    train-base)
        echo "Training ByT5-Base locally..."
        python -m src.cli.train_byt5 \
            --model_name google/byt5-base \
            --run_name byt5-base-v1 \
            --epochs 15 \
            --batch_size 4 \
            --gradient_accumulation 8 \
            --learning_rate 1e-4 \
            --include_lexicon \
            "${@:2}"
        ;;
        
    train-nllb)
        echo "Training NLLB-200 locally..."
        python -m src.cli.train_nllb \
            --run_name nllb-200-v1 \
            --epochs 10 \
            --batch_size 4 \
            --gradient_accumulation 8 \
            --learning_rate 1e-4 \
            --include_lexicon \
            "${@:2}"
        ;;
        
    train-qwen)
        echo "Training Qwen2.5-1.5B with DoRA locally..."
        python -m src.cli.train_qwen \
            --run_name qwen2-5-1_5b-v1 \
            --epochs 3 \
            --batch_size 2 \
            --gradient_accumulation 8 \
            "${@:2}"
        ;;
    
    infer)
        echo "Running local inference..."
        python -m src.cli.inference \
            --model_paths "models/byt5-small-v1/best_model" \
            --total_candidates 30 \
            --temperature 0.8 \
            "${@:2}"
        ;;
        
    evaluate)
        echo "Running local evaluation..."
        python -m src.cli.evaluate \
            --pred_file "${2:-submission.csv}" \
            --ref_file "${3:-data/sample_submission.csv}"
        ;;
        
    kaggle-dataset)
        MODEL_DIR="${2}"
        if [ -z "$MODEL_DIR" ]; then
            echo "Error: Must provide a model directory to upload (e.g. models/byt5-small-v1/best_model)"
            echo "Usage: ./run.sh kaggle-dataset path/to/model"
            exit 1
        fi
        
        DATASET_NAME=$(basename $(dirname "$MODEL_DIR"))
        DATASET_SLUG="deep-past-$DATASET_NAME"
        
        echo "Preparing Kaggle Dataset upload for $MODEL_DIR..."
        cd "$MODEL_DIR"
        
        cat > dataset-metadata.json << EOL
{
  "title": "$DATASET_SLUG",
  "id": "maazkhan711635/$DATASET_SLUG",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOL
        echo "Creating dataset on Kaggle..."
        kaggle datasets create -p . --dir-mode zip
        echo "Dataset upload complete!"
        ;;
    
    *)
        echo "Deep Past Challenge Pipeline - Local Environment"
        echo ""
        echo "Usage: ./run.sh <command> [args]"
        echo ""
        echo "Local Execution Commands:"
        echo "  preprocess       - Preprocess and align training data"
        echo "  train-small      - Train ByT5-Small"
        echo "  train-base       - Train ByT5-Base"
        echo "  train-nllb       - Train NLLB-200"
        echo "  train-qwen       - Train Qwen2.5-1.5B (DoRA)"
        echo "  infer            - Single model inference with MBR"
        echo "  evaluate         - Evaluate predictions against references"
        echo ""
        echo "Kaggle Tools:"
        echo "  kaggle-dataset   - Upload a local model directory as a Kaggle Dataset"
        ;;
esac
