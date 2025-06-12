# evaluate_with_metrics.py

import os
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spacy.tokens import DocBin
from sklearn.metrics import classification_report, confusion_matrix

# Load model and dev data
nlp = spacy.load("output/model-best")
doc_bin = DocBin().from_disk("data/dev_data.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

true_labels = []
pred_labels = []

for gold_doc in docs:
    pred_doc = nlp(gold_doc.text)

    # Map gold spans by text for alignment
    gold_ents = {(ent.start_char, ent.end_char): ent.label_ for ent in gold_doc.ents}
    pred_ents = {(ent.start_char, ent.end_char): ent.label_ for ent in pred_doc.ents}

    all_spans = set(gold_ents.keys()).union(set(pred_ents.keys()))
    for span in all_spans:
        true_labels.append(gold_ents.get(span, "O"))  # O = no entity
        pred_labels.append(pred_ents.get(span, "O"))

# Unique labels
labels = sorted(set(true_labels + pred_labels) - {"O"})

# Classification report
report = classification_report(true_labels, pred_labels, labels=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

os.makedirs("logs", exist_ok=True)
report_df.to_csv("logs/classification_report.csv")
print("Saved metrics to logs/classification_report.csv")

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/confusion_matrix.png")
print("Saved confusion matrix to plots/confusion_matrix.png")

