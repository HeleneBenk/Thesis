import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import ViTModel, AutoImageProcessor
from sklearn.preprocessing import MultiLabelBinarizer
import csv


MODEL_PATH = r"C:/Users/Helene/Desktop/Uni/Masterarbeit/ViT/Version 2/trial_9_model.pth"
CSV_PATH = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\Dataset\Label\Version 2\evalV2.csv"
IMAGE_DIR = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\Dataset\Gesplitteter Datensatz"
OUTPUT_CSV = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\ViT\Version 2\EvalOutput_trial9.csv"
THRESHOLD = 0.429194029
 


# === LABELS ===
material_labels = [
    "aluminium", "plastic", "metal", "residual waste", "cardboard",
    "organic waste", "composite carton", "paper", "brown glass",
    "green glass", "white glass", "pet", "rigid plastic container", "hazardous waste"
]
#disposal_labels = [
#    "yellow bin", "blue bin", "general household waste", "cardboard collection",
#    "organic waste bin", "paper collection", "brown glass collection point",
#    "green glass collection point", "white glass collection point",
#    "pet collection point", "battery collection point"
#]
all_labels = material_labels# + disposal_labels
#all_labels = disposal_labels
label_to_index = {label: i for i, label in enumerate(all_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}



def extract_combined_labels(label_str):
    binary = [0] * len(label_to_index)
    for label in label_str.lower().split("|"):
        label = label.strip()
        if label in label_to_index:
            binary[label_to_index[label]] = 1
    return binary

def decode(pred):
    return [index_to_label[i] for i, v in enumerate(pred) if v == 1]


class InferenceDataset(Dataset):
    def __init__(self, df, image_dir, processor, resize=(224, 224)):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Image ID"])
        image = Image.open(image_path).convert("RGB").resize(self.resize)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        labels = torch.tensor(extract_combined_labels(row["labels"]), dtype=torch.float)
        return pixel_values, labels, row["Image ID"]


class ViTMultiLabel(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.vit = base_model
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits


df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig", usecols=["Image ID", "labels"])
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
dataset = InferenceDataset(df, IMAGE_DIR, processor)
dataloader = DataLoader(dataset, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model = ViTMultiLabel(base_model, num_labels=len(all_labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("CSV-Spalten:", df.columns.tolist())
print("Beispiel-Labels:", df["labels"].head(3).tolist())


results = []
with torch.no_grad():
    for pixel_values, labels, image_id in dataloader:
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > THRESHOLD).astype(int)
        results.append({
            "image_id": image_id[0],
            "true_labels": "|".join(decode(labels[0].numpy())),
            "predicted_labels": "|".join(decode(preds))
        })


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "true_labels", "predicted_labels"], delimiter=";")
    writer.writeheader()
    writer.writerows(results)

print(f"Ergebnisse gespeichert in: {OUTPUT_CSV}")




