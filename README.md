# Masters Thesis – Multimodal Waste Classification with YOLO, ViLT, CuMo, and ViT

This repository contains the complete code, evaluation results, and documentation for my Master's Thesis:

> **Don’t waste Compute Efficient AI Models for Waste Management**  
> Goal: To evaluate the effectiveness of multimodal transformer models in recognizing and classifying waste items for improved recycling recommendations in a household setting.

---

##Repository Structure

```bash
Thesis/
├── Cumo/               # Fine-tuning scripts and evaluation for CuMo
├── Misc_Code/          # Helper scripts: evaluation, augmentation, label creation
├── ViLT/               # Fine-tuning and Optuna tuning for ViLT, Dataset, best Models and Results
├── ViT/                # Fine-tuning and Optuna tuning for ViT, Dataset, best Models and Results
├── YOLO/               # Object detection with YOLO11
├── PictureIDs_Label_Source.xlsx  # Mapping Pictures to their Source
├── Poster_ThesisBenkert.pdf      # Poster version of the thesis project
├── .gitattributes       # LFS management for large model files
└── README.md            # You're here!
