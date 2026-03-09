# Repository Organization Plan

The repository currently contains a mix of local CLI scripts, robust standalone Kaggle notebooks (nb1-nb5), and obsolete string-generator scripts that were used to stitch Python strings together into Kaggle submissions. I propose a clean, standard ML engineering project structure.

## Proposed Changes

### 1. Folder Restructuring
We will split the `src/` directory into two distinct, purpose-built folders:

#### `src/cli/` (Local Runners)
These are intended to run on your local machine or an unconstrained VM. They take command-line arguments and assume full filesystem access.
- `preprocess.py`
- `train_byt5.py`
- `train_nllb.py`
- `train_qwen.py` (Local Qwen fine-tuning with DoRA)
- `inference.py`
- `evaluate.py`

#### `src/kaggle/` (Kaggle Notebooks)
These are 100% standalone scripts. You just copy the contents of the file, paste it into a blank Kaggle Notebook, and press "Run All". They have no local dependencies.
- `nb1_train_byt5_small.py` (formerly `kaggle_nb1_byt5_small.py`)
- `nb2_train_byt5_base.py` (formerly `kaggle_nb2_byt5_base.py`)
- `nb3_train_nllb.py` (formerly `kaggle_nb3_nllb.py`)
- `nb5_train_qwen.py` (formerly `kaggle_nb5_qwen.py`)
- `nb4_ensemble_inference.py` (formerly `kaggle_nb4_inference.py`)

### 2. Cutting the Bloat [DELETE]
We will permanently delete the following obsolete files. These were intermediate "generator" scripts that stitched Python code together as strings. Now that we have directly written the standalone `kaggle_nb*.py` scripts, these generators are pure bloat and completely break the code structure.

- `src/kaggle_train_notebook.py`
- `src/kaggle_train_nllb.py`
- `src/kaggle_notebook.py`
- `src/kaggle_sweep_notebook.py`
- `src/kaggle_interactive_runner.py` (The massive 22KB monolithic file we abandoned)
- And their corresponding generated files in the root directory: `kaggle_submission.py`, `kaggle_train_notebook.py`, etc.

### 3. Updating `run.sh` [MODIFY]
- Update `run.sh` to point to the new `src/cli/` paths.
- Remove the obsolete generation commands (e.g., `kaggle-train`, `kaggle-submit`) that relied on the deleted generator scripts.

## Verification Plan
1. Move the files to their respective `src/cli/` and `src/kaggle/` directories.
2. Delete the obsolete scripts.
3. Update `run.sh`.
4. Perform a local dry-run of a CLI tool via `run.sh` (e.g., `./run.sh evaluate`) to ensure the references are correct.
