import pandas as pd
import json
import os


csv_path = r"C:/Users/Helene/Desktop/Uni/Masterarbeit/Dataset/Label/Train_Val_Eval/eval_Truth_V3.csv" #csv with true labels/Prompts
jsonl_dir = r"C:/Users/Helene/Desktop/Uni/Masterarbeit/Cumo/Version 3"
mapping_path = r"C:/Users/Helene/Desktop/Uni/Masterarbeit/Cumo/Version 3/questionV3.jsonl"
output_dir = r"C:\Users\Helene\Desktop\Uni\Masterarbeit\Cumo\Version 3"

#Mapping: question_id and image_id
with open(mapping_path, encoding="utf-8") as f:
    mapping_data = [json.loads(line) for line in f]
mapping_df = pd.DataFrame(mapping_data)
mapping_df["image_id"] = mapping_df["image"].str.replace(".jpg", "", regex=False)

#Label
df_labels = pd.read_csv(csv_path, sep=";", usecols=["Image ID", "labels"])
df_labels.columns = ["image", "true_labels"]
df_labels["image_id"] = df_labels["image"].str.replace(".jpg", "", regex=False)

#Loop all outputX.jsonl
for filename in os.listdir(jsonl_dir):
    if filename.startswith("output") and filename.endswith(".jsonl"):
        file_path = os.path.join(jsonl_dir, filename)

        
        all_preds = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_preds.append({
                    "question_id": obj["question_id"],
                    "predicted_labels": obj["text"]
                })

        df_preds = pd.DataFrame(all_preds)

        #Mapping
        df_preds = df_preds.merge(mapping_df[["question_id", "image_id"]], on="question_id", how="left")
        df_merged = df_preds.merge(df_labels[["image_id", "true_labels"]], on="image_id", how="left")
        df_merged = df_merged[["image_id", "true_labels", "predicted_labels"]].dropna()

        
        suffix = filename.replace("output", "").replace(".jsonl", "")
        output_file = os.path.join(output_dir, f"merged_predictions{suffix}.csv")
        df_merged.to_csv(output_file, sep=";", index=False)

print(pd.read_csv(csv_path, sep=";").columns)