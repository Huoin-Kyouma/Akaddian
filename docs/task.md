# Deep Past Challenge - Task Tracking

## Phase 3: Model Diversity
- [/] Train ByT5-Base <!-- id: 0 -->
    - [x] Modify `kaggle_train_notebook.py` to accept arguments or duplicate it for Base <!-- id: 1 -->
    - [x] Update Kaggle Kernel push logic to support multiple models (`byt5-base`) <!-- id: 2 -->
    - [ ] Wait for GPU Quota to Push ByT5-Base to Kaggle and verify training <!-- id: 3 -->
- [/] Fine-tune NLLB-200 with Akkadian token <!-- id: 4 -->
    - [x] Research NLLB-200 tokenization and vocabulary manipulation <!-- id: 5 -->
    - [x] Develop `src/train_nllb.py` <!-- id: 6 -->
    - [x] Create Kaggle Notebook generator `kaggle_train_nllb.py` <!-- id: 7 -->
    - [ ] Wait for GPU Quota to Push NLLB to Kaggle and verify training <!-- id: 8 -->
- [/] Hyperparameter sweep <!-- id: 9 -->
    - [x] Plan sweep strategy (LR, Dropout, Weight Decay) <!-- id: 10 -->
    - [x] Implement W&B logging and Kaggle sweep notebook generator <!-- id: 11 -->
    - [ ] Wait for GPU Quota to Push Sweep to Kaggle and analyze results <!-- id: 12 -->

## Phase 4: MBR & Ensemble Optimization
- [x] Single-model MBR with chrF++ <!-- id: 13 -->
    - [x] Core algorithmic MBR sequence implemented in `inference.py` <!-- id: 14 -->
    - [x] Vectorize `compute_chrf_matrix` or optimize using multiprocessing (replaced with pytorch batch generation) <!-- id: 15 -->
- [x] Cross-model candidate pooling <!-- id: 16 -->
    - [x] Basic multi-model configuration string structured in `kaggle_notebook.py` <!-- id: 17 -->
    - [x] Refactor Kaggle Notebook to generate candidates in batched tensors to prevent 9h timeout <!-- id: 18 -->
- [x] Named Entity post-processing <!-- id: 19 -->
    - [x] Load `named_entities.json` dictionary (or dynamically generate on Kaggle) <!-- id: 20 -->
    - [x] Implement string matching to enforce valid Entity translation during inference <!-- id: 21 -->

## Phase 5: Polish & Submit
- [x] Automate Model Uploads to Kaggle Datasets <!-- id: 22 -->
    - [x] Create `kaggle_dataset.py` or script to zip local models and push as Kaggle Dataset <!-- id: 23 -->
- [x] Final Validation Review <!-- id: 24 -->
    - [x] Ensure CV vs LB consistency code is clean in `evaluate.py` <!-- id: 25 -->
- [x] Submit Final Submissions <!-- id: 26 -->
    - [x] Add `kaggle-submit` command to `run.sh` to trigger the final `kaggle_submission.py` notebook <!-- id: 27 -->

## Phase 6: Repository Organization and Cleanup
- [x] Refactor directory structure to separate CLI environments from Kaggle environments <!-- id: 28 -->
    - [x] Create `src/cli/` and move local training/evaluation scripts in <!-- id: 29 -->
    - [x] Create `src/kaggle/` and move standalone `kaggle_nb*.py` scripts in <!-- id: 30 -->
- [x] Remove bloat and obsolete scripts <!-- id: 31 -->
    - [x] Delete `kaggle_interactive_runner` and the `kaggle_*_notebook.py` text-generator scripts <!-- id: 32 -->
- [x] Verify `run.sh` operates securely against the new paths without breaking <!-- id: 33 -->
