# YOLOv11 – Master Thesis

This directory contains all relevant components for the training, evaluation, and deployment of the YOLOv11 object detection model used in the master thesis.

## Contents

- **Best Model/**  
  Contains the final YOLOv11 model checkpoint (`best.pt`) selected based on evaluation performance.

- **ScrumScripts/**  
  Previously used scripts for batch training via SLURM or local automation (now removed).

- **train/**  
  The training dataset including images and labels in YOLO format.

- **val/**  
  The validation dataset used during model training for performance monitoring.

- **test/**  
  The test dataset used for final model evaluation.

- **dataset.yaml**  
  Configuration file describing the dataset structure (classes, paths, splits).

- **requirements.yml**  
  Conda environment file listing all dependencies required to run the YOLOv11 training and evaluation pipeline.


