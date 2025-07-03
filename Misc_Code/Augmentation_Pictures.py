import os
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path

csv_path = Path("C:/Users/Helene/Desktop/Uni/Masterarbeit/Dataset/Label/Label.csv")
image_base_path = Path("C:/Users/Helene/Desktop/Uni/Masterarbeit/Dataset/Fertignummeriert")
output_path = Path("C:/Users/Helene/Desktop/Uni/Masterarbeit/Dataset/Augmented_Images")
output_path.mkdir(parents=True, exist_ok=True)

start_id = 5048  # First new ID for augmented images
target_image_count = 300  # Target number of images per class

df = pd.read_csv(csv_path, sep=';')

#Identify categories with too few images
category_counts = df['Kategorie'].value_counts()
underrepresented = category_counts[category_counts < target_image_count].to_dict()

def strong_augment(img):
    if random.random() > 0.5:
        img = TF.hflip(img)
    angle = random.choice([0, 90, 180, 270])
    img = TF.rotate(img, angle)
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    )
    img = color_jitter(img)
    return img


augmented_entries = []
current_id = start_id


for category, count in underrepresented.items():
    class_df = df[df['Kategorie'] == category]
    image_paths = [
        image_base_path / f"{row['ID']}{row['Endung']}"
        for _, row in class_df.iterrows()
    ]

    needed = target_image_count - count
    for i in range(needed):
        original_path = random.choice(image_paths)
        if original_path.exists():
            try:
                with Image.open(original_path) as img:
                    img_aug = strong_augment(img)
                    new_filename = f"{current_id}.jpg"
                    save_path = output_path / new_filename
                    img_aug.save(save_path)

                    augmented_entries.append({
                        "ID": current_id,
                        "OriginalID": original_path.stem,
                        "Category": category
                    })

                    current_id += 1
            except Exception as e:
                print(f"Error with {original_path.name}: {e}")

aug_df = pd.DataFrame(augmented_entries)
aug_df.to_csv(output_path / "augmented_labels.csv", sep=';', index=False)
print(f"Augmentation completed. {len(aug_df)} new images saved.")
