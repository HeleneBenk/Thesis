import os
# === ENVIRONMENT SETTINGS ===
os.environ["HF_HOME"] = "/cfs/earth/scratch/benkehel/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/cfs/earth/scratch/benkehel/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/cfs/earth/scratch/benkehel/huggingface"
os.environ["HF_METRICS_CACHE"] = "/cfs/earth/scratch/benkehel/huggingface"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel, AutoImageProcessor
from sklearn.metrics import f1_score
import numpy as np
import re
import optuna
import csv



IMAGE_DIR = "/cfs/earth/scratch/benkehel/ViLT/Data/Pictures"
TRAIN_JSON = "/cfs/earth/scratch/benkehel/ViLT/Data/trainV3_ViLT.json"
VAL_JSON = "/cfs/earth/scratch/benkehel/ViLT/Data/valV3_ViLT.json"
LOG_DIR = "/cfs/earth/scratch/benkehel/ViT/V3/optuna_logsV3"
os.makedirs(LOG_DIR, exist_ok=True)


disposal_labels = [
    "yellow bin", "blue bin", "general household waste", "cardboard collection",
    "organic waste bin", "paper collection", "brown glass collection point",
    "green glass collection point", "white glass collection point",
    "pet collection point", "battery collection point"
]

label_to_index = {label: idx for idx, label in enumerate(disposal_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

def extract_disposal_from_gpt(text):
    labels = [0] * len(disposal_labels)
    items = [t.strip() for t in text.lower().split(",")]
    for item in items:
        if item in label_to_index:
            labels[label_to_index[item]] = 1
    return labels




def load_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [{
        "image_file": item["image"],
        "labels": extract_disposal_from_gpt(item["conversations"][1]["value"])
    } for item in data]


train_data = load_json(TRAIN_JSON)
val_data = load_json(VAL_JSON)

#dataset
class WasteDatasetVit(Dataset):
    def __init__(self, data, feature_extractor, image_dir, resize=(224, 224)):
        self.data = data
        self.feature_extractor = feature_extractor
        self.image_dir = image_dir
        self.resize = resize

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.image_dir, item["image_file"])).convert("RGB").resize(self.resize)
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(item["labels"], dtype=torch.float)
        inputs["image_file"] = item["image_file"]
        return inputs

def collate_fn(batch):
    return {k: [item[k] for item in batch] if k == "image_file" else torch.stack([item[k] for item in batch]) for k in batch[0]}

#Model
class ViTMultiLabel(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.vit = base_model
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        loss = F.binary_cross_entropy_with_logits(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

def decode(preds):
    return [index_to_label[i] for i, val in enumerate(preds) if val == 1]

#optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    threshold = trial.suggest_float("threshold", 0.2, 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model + Processor
    base_model = ViTModel.from_pretrained(
        "google/vit-base-patch16-224", cache_dir="/cfs/earth/scratch/benkehel/huggingface"
    )
    model = ViTMultiLabel(base_model, len(disposal_labels)).to(device)

    feature_extractor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224", cache_dir="/cfs/earth/scratch/benkehel/huggingface"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(WasteDatasetVit(train_data, feature_extractor, IMAGE_DIR), batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(WasteDatasetVit(val_data, feature_extractor, IMAGE_DIR), batch_size=4, shuffle=False, collate_fn=collate_fn)

    best_val_loss = float("inf")
    no_improvement = 0
    training_losses = []
    validation_losses = []

    for epoch in range(10):
        #Train
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**input_batch)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += out["loss"].item()
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        #eval
        model.eval()
        preds, trues, filenames = [], [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                input_batch = {k: v for k, v in batch_cuda.items() if isinstance(v, torch.Tensor)}
                out = model(**input_batch)
                logits = out["logits"].cpu()
                total_val_loss += out["loss"].item()
                probs = torch.sigmoid(logits).numpy()
                preds.extend((probs > threshold).astype(int))
                trues.extend(batch["labels"].numpy())
                filenames.extend(batch["image_file"])
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        f1 = f1_score(trues, preds, average="weighted")
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 2:
                break


    model_path = os.path.join(LOG_DIR, f"trial_{trial.number}_modelViTV3.pth")
    torch.save(model.state_dict(), model_path)
    csv_path = os.path.join(LOG_DIR, f"trial_{trial.number}_predictionsViTV3.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Image", "True Labels", "Predicted Labels"])
        for i in range(len(preds)):
            writer.writerow([
                filenames[i],
                "|".join(decode(trues[i])),
                "|".join(decode(preds[i]))
            ])

    log_csv = os.path.join(LOG_DIR, "optuna_trial_logViTV3.csv")
    header = not os.path.exists(log_csv)
    with open(log_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["Trial", "lr", "threshold", "f1", "val_loss", "model_path", "csv_path"])
        writer.writerow([trial.number, lr, threshold, f1, best_val_loss, model_path, csv_path])

    return f1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(f"  F1: {study.best_trial.value}")
for k, v in study.best_trial.params.items():
    print(f"    {k}: {v}")
