# Kaggle Integration Walkthrough

To move training directly to Kaggle's free GPUs while bypassing the need to upload large local datasets, I implemented a custom Kaggle Notebook generator.

## What changed?

1.  **Removed Local Preprocessing Overhead**: We no longer produce the `processed/` directory locally or try to upload it as a Kaggle Dataset.
2.  **`src/kaggle_train_notebook.py`**: I created a script that generates a standalone Python file (`kaggle_train_notebook.py`). This script:
    *   Imports all the data preprocessing functions you built in Phase 1 (sentence alignment, cleaning, taking the validation split, and incorporating eBL dictionary lexicon pairs).
    *   Uses the competition dataset (`/kaggle/input/deep-past-initiative-machine-translation`) which is already available on Kaggle.
    *   Directly feeds the processed datasets into the HuggingFace `Seq2SeqTrainer` to fine-tune the `ByT5-Small` model.
    *   Zips up the final `best_model` folder in `/kaggle/working/` so you can easily download it from the output tab.
3.  **`run.sh` Kaggle Push**: I added a `kaggle-push` command to the main script. Running `./run.sh kaggle-push` generates the notebook, builds the Kaggle Kernel metadata, resolving it to the competition dataset source, and automatically pushes it to Kaggle to start execution immediately. 

## Validation Results

The training notebook was successfully pushed to Kaggle using your API key. 

You can monitor the live training progress at:
**[https://www.kaggle.com/maazkhan711635/deep-past-byt5-training](https://www.kaggle.com/maazkhan711635/deep-past-byt5-training)**

Once the notebook finishes executing, the `best_model` will be available in the "Output" section of that Kaggle Notebook page, ready for you to download and drop into your local `models/` directory for Phase 4 inference.

## Phase 3: Model Diversity Implementation

Phase 3 introduces tools to build a diverse model ensemble and optimize hyperparameters. Because Kaggle free-tier restricts users to **2 concurrent GPU sessions**, these jobs must be launched sequentially once quota opens.

### 1. ByT5-Base Parameterization
- **Change:** The Kaggle Notebook generator (`src/kaggle_train_notebook.py`) and the `run.sh` script were parameterized to accept the model size (`small` or `base`). Batch sizes and gradient accumulation steps adjust dynamically to fit GPU RAM.
- **Execution:** Wait for GPU quota, then run:
  ```bash
  ./run.sh kaggle-push base
  ```

### 2. NLLB-200 Integration
- **Change:** Created a dedicated Kaggle deployment script (`src/kaggle_train_nllb.py`) for the `facebook/nllb-200-distilled-600M` model. This script adds a custom Akkadian token (`akk_Latn`) to the tokenizer, resizes model embeddings, and explicitly defines the source/target mappings during training prep.
- **Execution:** Wait for GPU quota, then deploy using:
  ```bash
  ./run.sh kaggle-push nllb
  ```

### 3. Hyperparameter Sweeps (Weights & Biases)
- **Change:** Created `src/kaggle_sweep_notebook.py` to deploy a Kaggle notebook that executes an automatic Weights & Biases (W&B) Bayesian sweep across learning rates, dropout, and weight decay to find the optimal configuration for your dataset.
- **Setup:** You must add a Kaggle Secret named `WANDB_API_KEY` to your Kaggle account before running this.
- **Execution:** Deploy the sweep notebook using:
  ```bash
  ./run.sh kaggle-push sweep
  ```

## Phase 4: MBR & Ensemble Optimization

Phase 4 bridges the gap between raw model outputs and the final Kaggle submission by optimizing the mathematical inference process. The main goal was to speed up inference to fit within Kaggle's **9-hour timeout** limit, while leveraging Ensembling and Named Entities.

### 1. Batched GPU Generation
- **Change:** `src/inference.py` and the `kaggle_notebook` generator were completely rewritten to use batched tensor generation via PyTorch, replacing the previous sentence-by-sentence loop. 
- **Impact:** Grouping inputs into dynamically padding batches of 16 significantly cuts down generation time, ensuring that generating 30 candidates per sentence per model will not cause Kaggle notebooks to time out before MBR finishes decoding.

### 2. Named Entity Post-Processing
- **Change:** A new algorithmic filter, `post_process()`, runs automatically after MBR decoding selects the best candidate. 
- **How it works:** It dynamically reads `published_texts.csv` during execution, applying Regex to rebuild the list of valid Akkadian Named Entities (e.g. counts `KIŠIB` names). If it detects a known noun transliteration in the source text, it translates the raw unaccented, hyphenated string into proper English Title Case, scanning the output to fix any casing errors the model makes.

## Phase 5: Polish & Submit

With all Code & Model logic finished, the final obstacle is deploying this pipeline for Kaggle's offline grading. You must "attach" your locally trained weights as Datasets, as the competition inference kernels cannot download from HuggingFace directly.

### 1. Upload Model Weights to Kaggle
- **Change:** I added a `kaggle-dataset` command to `run.sh`. This automatically builds the Kaggle metadata for a given local folder, ZIPs it, and pushes it to Kaggle as a private Dataset containing your `.bin` model weights.
- **Execution:** Once you have a trained model locally (e.g., from downloading the `best_model` output from your Kaggle training notebook), run:
  ```bash
  ./run.sh kaggle-dataset models/byt5-small-v1/best_model
  ```
  *(Repeat this for your base and nllb models when they finish)*

### 2. Execute the Final Submission
- **Change:** I added a `kaggle-submit` command. This command triggers the `kaggle_notebook.py` generation template (which now includes the Batched PyTorch generation from Phase 4), bundles it with a `kernel-metadata.json`, and pushes it to the competition server.
- **Note:** The metadata is currently hard-coded to attach the dataset `maazkhan711635/deep-past-byt5-small-v1` (which you will create in step 1). If you ensemble more models, add their dataset slugs to the `dataset_sources` array in `run.sh` line 155.
- **Execution:**
  ```bash
  ./run.sh kaggle-submit
  ```
  *(This kernel is set to `enable_internet: false` per competition rules. Once it finishes executing on Kaggle, simply hit the "Submit to Competition" button on the output `submission.csv`!)*

---

## Bypassing the Kaggle Queue (Interactive Workaround)

If the Kaggle CLI is placing your pushes into an endless queue because background tasks are given lower priority for GPU allocation, I have created a monolithic bypass file: `src/kaggle_interactive_runner.py`.

This single Python file contains the logic for **Phases 1 through 5** completely smashed together, separated by clear `# --- CELL ---` blocks.

**How to use:**
1. Open a new Kaggle Interactive Notebook and ensure your `deep-past-initiative-machine-translation` dataset is attached.
2. Open `src/kaggle_interactive_runner.py` locally.
3. Copy the contents of `CELL 1` and paste it into the first code cell on Kaggle. Run it.
4. Copy `CELL 2` into the next Kaggle block. Run it. 
5. Continue this down to `CELL 5`. This effectively runs the ENTIRE system—from preprocessing to metric evaluation and valid `.csv` submission generation—in a single, high-priority interactive GPU session, completely bypassing the CLI's wait times!
