import os
import spacy
from spacy.tokens import DocBin
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# === Ensure output folders ===
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# === Load spaCy model & training data ===
nlp = spacy.blank("de")
doc_bin = DocBin().from_disk("data/train_data.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# === Count occurrences ===
label_text_counter = defaultdict(Counter)
for doc in docs:
    for ent in doc.ents:
        label_text_counter[ent.label_][ent.text.strip()] += 1

# === Log results ===
with open("logs/entity_distribution.log", "w", encoding="utf-8") as f:
    f.write("Entity Text Frequency per Label\n\n")
    for label, text_counter in label_text_counter.items():
        f.write(f"Label: {label}\n")
        for text, count in text_counter.most_common():
            f.write(f"   {text}: {count}\n")
        f.write("\n")

print("Logged entity distribution to logs/entity_distribution.log")

# === Plot top N per label ===
top_n = 10
for label, text_counter in label_text_counter.items():
    most_common = text_counter.most_common(top_n)
    if not most_common:
        continue

    texts, counts = zip(*most_common)

    plt.figure(figsize=(10, 5))
    plt.barh(texts, counts)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} frequent values for '{label}'")
    plt.xlabel("Count")
    plt.tight_layout()
    filename = f"plots/{label}_top_{top_n}.png"
    plt.savefig(filename)
    plt.close()

print("Plots saved to plots/")

