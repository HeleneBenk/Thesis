# ViT Models

This directory contains the full workflow related to ViT models developed for the masters thesis, including data, model training, evaluation, and inference.

## Contents

- **V1/**, **V2/**, **V3/**  
  Each folder contains a different version of the ViLT model.  
  Each version includes:
  - Two best model checkpoints (`.pt`)
  - Evaluation output for the two best models as `.csv`
  - Evaluation metrics as `.txt`
  - Training code

- **Data/**  
  Pictures and json/csv files used for training and evaluation.  
  **Note**: The same dataset structure and file names as used for ViLT were reused here.

- **Scrum Input HPC/**  
  Prepared SLURM and training scripts used for high-performance computing environments.

- **InferenceOnEvalsetViLT.py**  
  Script to run inference on the evaluation set using the trained ViLT models.
