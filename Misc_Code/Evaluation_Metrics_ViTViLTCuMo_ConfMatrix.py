import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, multilabel_confusion_matrix
import os

output_dir = "C:/Users/Helene/Desktop/Uni/Masterarbeit/Cumo/Version 1"
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(
    os.path.join(output_dir, 'merged_predictionsV1.csv'),
    sep=';',
    names=['image_id', 'true_labels', 'predicted_labels'],
    header=1
)


def clean_and_split(x):
    return [label.strip() for label in x.split('|') if label.strip()] if isinstance(x, str) and x.strip() else []

df['true_labels'] = df['true_labels'].apply(clean_and_split)
df['predicted_labels'] = df['predicted_labels'].apply(clean_and_split)


mlb = MultiLabelBinarizer()
true_labels = mlb.fit_transform(df['true_labels'])
predicted_labels = mlb.transform(df['predicted_labels'])
label_names = mlb.classes_
n_labels = len(label_names)


print("Hamming Loss:", hamming_loss(true_labels, predicted_labels))
print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=label_names, zero_division=0))

print(multilabel_confusion_matrix(true_labels, predicted_labels))


conf_matrix = np.zeros((n_labels, n_labels), dtype=int)

for t, p in zip(df['true_labels'], df['predicted_labels']):
    true_set = set(t)
    pred_set = set(p)

    for label in true_set & pred_set:
        i = list(label_names).index(label)
        conf_matrix[i, i] += 1

    for label in pred_set - true_set:
        j = list(label_names).index(label)
        for source in true_set:
            i = list(label_names).index(source)
            conf_matrix[i, j] += 1

conf_df = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
#print("\nStrict Label Confusion Matrix:")
#print(conf_df)


fontname = 'Arial'

plt.figure(figsize=(14, 10))
sns.heatmap(conf_df,
            annot=True,
            fmt='d',
            cmap="rocket_r",
            vmin=0,
            vmax=conf_df.values.max(),
            annot_kws={"size": 16, "fontname": fontname})

plt.xlabel("Predicted Label", fontsize=14, fontname=fontname)
plt.ylabel("True Label", fontsize=14, fontname=fontname)
plt.xticks(rotation=45, ha='right', fontsize=16, fontname=fontname)
plt.yticks(rotation=0, fontsize=16, fontname=fontname)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "strict_confusion_matrix_ViTV3.pdf"), bbox_inches='tight')
plt.show()


underpredicted_counts = np.sum((true_labels == 1) & (predicted_labels == 0), axis=0)


empty_preds = df['predicted_labels'].apply(lambda x: not x)
missed_when_empty_dict = {label: 0 for label in label_names}

for true, pred in zip(df['true_labels'], df['predicted_labels']):
    if not pred:  
        for label in true:
            missed_when_empty_dict[label] += 1


missed_when_empty = [missed_when_empty_dict[label] for label in label_names]


under_df = pd.DataFrame({
    'Label': label_names,
    'Missed Predictions (False Negatives) (incl. NaNs):': underpredicted_counts,
    'NaNs': missed_when_empty
}).sort_values(by='Missed Predictions (False Negatives) (incl. NaNs):', ascending=False)

print("\nMissed Predictions incl. NaNs:")
print(under_df.to_string(index=False))


#under_df.to_csv(os.path.join(output_dir, "underpredictions_with_empty.csv"), index=False)


total_empty_preds = empty_preds.sum()
print(f"\nTotal NaNs (pred == []): {total_empty_preds}")
