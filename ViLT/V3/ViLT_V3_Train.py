import os

# Environment for HF cache
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
from transformers import ViltProcessor, ViltModel
from sklearn.metrics import f1_score
import numpy as np
import re
import optuna
import csv


IMAGE_DIR = "/cfs/earth/scratch/benkehel/ViLT/Data/Pictures"
TRAIN_JSON = "/cfs/earth/scratch/benkehel/ViLT/Data/trainV3_ViLT.json"
VAL_JSON = "/cfs/earth/scratch/benkehel/ViLT/Data/valV3_ViLT.json"
LOG_DIR = "/cfs/earth/scratch/benkehel/ViLT/V3/optuna_logs/V3"
os.makedirs(LOG_DIR, exist_ok=True)

disposal_labels = [
    "yellow bin", "blue bin", "general household waste", "cardboard collection",
    "organic waste bin", "paper collection", "brown glass collection point",
    "green glass collection point", "white glass collection point",
    "pet collection point", "battery collection point"
]


all_labels = disposal_labels
label_to_index = {label: idx for idx, label in enumerate(all_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}


def extract_combined_labels(text):
    labels = [0] * len(all_labels)
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
        "question": item["conversations"][0]["value"],
        "labels": extract_combined_labels(item["conversations"][1]["value"])
    } for item in data]

train_data = load_json(TRAIN_JSON)
val_data = load_json(VAL_JSON)

class WasteDataset(Dataset):
    def __init__(self, data, processor, image_dir, resize=(224, 224)):
        self.data = data
        self.processor = processor
        self.image_dir = image_dir
        self.resize = resize

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.image_dir, item["image_file"])).convert("RGB").resize(self.resize)
        inputs = self.processor(images=img, text=item["question"], return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(item["labels"], dtype=torch.float)
        inputs["image_file"] = item["image_file"]
        return inputs

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: [item[k] for item in batch] if k == "image_file" else torch.stack([item[k] for item in batch]) for k in keys}

class ViltMultiLabel(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.vilt = base_model
        self.classifier = nn.Linear(self.vilt.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        outputs = self.vilt(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            token_type_ids=kwargs["token_type_ids"],
            pixel_values=kwargs["pixel_values"],
            pixel_mask=kwargs["pixel_mask"]
        )
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        loss = None
        if "labels" in kwargs:
            loss = F.binary_cross_entropy_with_logits(logits, kwargs["labels"])
        return {"logits": logits, "loss": loss}

def decode(preds):
    return [index_to_label[i] for i, val in enumerate(preds) if val == 1]

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    threshold = trial.suggest_float("threshold", 0.2, 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltMultiLabel(ViltModel.from_pretrained("dandelin/vilt-b32-mlm"), num_labels=len(all_labels)).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(WasteDataset(train_data, processor, IMAGE_DIR), batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(WasteDataset(val_data, processor, IMAGE_DIR), batch_size=4, shuffle=False, collate_fn=collate_fn)

    best_val_loss = float("inf")
    training_losses = []
    validation_losses = []
    epochs_without_improvement = 0

    for epoch in range(10):
        #Training
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(**batch)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += out["loss"].item()
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        #Validation
        model.eval()
        preds, trues, filenames = [], [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(**batch_cuda)
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 2:
                break

    #Save model
    model_path = os.path.join(LOG_DIR, f"trial_{trial.number}_model.pth")
    torch.save(model.state_dict(), model_path)

    #Save predictions
    csv_path = os.path.join(LOG_DIR, f"trial_{trial.number}_predictions.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Image", "True Labels", "Predicted Labels"])
        for i in range(len(preds)):
            writer.writerow([
                filenames[i],
                "|".join(decode(trues[i])),
                "|".join(decode(preds[i]))
            ])

    #Loss Curve
    loss_csv_path = os.path.join(LOG_DIR, f"trial_{trial.number}_losses.csv")
    with open(loss_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        for i in range(len(training_losses)):
            writer.writerow([i + 1, training_losses[i], validation_losses[i]])

    #Optuna
    log_csv = os.path.join(LOG_DIR, "optuna_trial_log.csv")
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
