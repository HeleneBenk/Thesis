import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import ViltProcessor, ViltModel
import csv


MODEL_PATH = r"C:/Users/Helene/Desktop/Uni/Masterarbeit/ViT/Version 2/trial_1_model.pth"
CSV_PATH = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\Dataset\Label\Version 2\evalV2.csv"
IMAGE_DIR = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\Dataset\Pictures"
OUTPUT_CSV = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\ViT\Version 2\EvalOutput_trial1.csv"
THRESHOLD = 0.429194029




# Labels - depend on V1, V2 or V3
material_labels = [
    "aluminium", "plastic", "metal", "residual waste", "cardboard",
    "organic waste", "composite carton", "paper", "brown glass",
    "green glass", "white glass", "pet", "rigid plastic container", "hazardous waste"
]
disposal_labels = [
    "yellow bin", "blue bin", "general household waste", "cardboard collection",
    "organic waste bin", "paper collection", "brown glass collection point",
    "green glass collection point", "white glass collection point",
    "pet collection point", "battery collection point"
]

#all_labels = material_labels + disposal_labels
all_labels = disposal_labels
label_to_index = {label: i for i, label in enumerate(all_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}


# Depends on V1, V2 or V3
FIXED_TEXT = "How to dispose of this item?"


def extract_combined_labels(label_str):
    binary = [0] * len(label_to_index)
    for label in label_str.lower().split("|"):
        label = label.strip()
        if label in label_to_index:
            binary[label_to_index[label]] = 1
    return binary

def decode(pred):
    return [index_to_label[i] for i, v in enumerate(pred) if v == 1]


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
        return logits



class InferenceDatasetVilt(Dataset):
    def __init__(self, df, image_dir, processor, fixed_text, resize=(224, 224)):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.fixed_text = fixed_text
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Image ID"])
        image = Image.open(image_path).convert("RGB").resize(self.resize)
        labels = torch.tensor(extract_combined_labels(row["labels"]), dtype=torch.float)
        encoding = self.processor(image, self.fixed_text, return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.squeeze(0) for k, v in encoding.items()}
        return inputs, labels, row["Image ID"]


df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig", usecols=["Image ID", "labels"])
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
dataset = InferenceDatasetVilt(df, IMAGE_DIR, processor, FIXED_TEXT)
dataloader = DataLoader(dataset, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViltMultiLabel(ViltModel.from_pretrained("dandelin/vilt-b32-mlm"), num_labels=len(all_labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


results = []
with torch.no_grad():
    for batch, labels, image_id in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)

        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > THRESHOLD).astype(int)
        results.append({
            "image_id": image_id,
            "true_labels": "|".join(decode(labels.squeeze(0).numpy())),
            "predicted_labels": "|".join(decode(preds))
        })



with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "true_labels", "predicted_labels"], delimiter=";")
    writer.writeheader()
    writer.writerows(results)

print(f"Saved predictions to: {OUTPUT_CSV}")
