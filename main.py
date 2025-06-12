import spacy
import pandas as pd
import re
import pickle
import string
from sklearn.model_selection import train_test_split
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
from spacy.tokens import DocBin
from spacy.util import filter_spans

# Load dataset
df = pd.read_csv("data/ds_ner_test_case.csv")

# Load spaCy German model
nlp = spacy.load("de_core_news_sm")

# Clean text: remove HTML, punctuation, extra spaces
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

# Add EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")

# COLOR patterns using regex
color_patterns = [
    {"label": "COLOR", "pattern": [{"TEXT": {"REGEX": "(?i)^" + color + ".*"}}]}
    for color in [
        "schwarz", "weiß", "rot", "blau", "grün", "gelb", "grau", "silber", "gold",
        "beige", "braun", "violett", "lila", "orange", "pink", "magenta", "cyan",
        "mattschwarz", "anthrazit", "kupfer"
    ]
]

# QUALITY adjective roots with regex
adjectives = [
    "schnell", "hochgeschwindigkeit", "rasch", "ultraschnell", "leistungsstark", "effizient",
    "sauber", "klar", "strukturiert", "scharf", "poliert", "glatt",
    "leistungsfähig", "kraftvoll", "superschnell", "fantastisch", "großartig", "hochwertig", "qualitativ"
]
adjective_regex = "(?i)^(" + "|".join(adj + ".*" for adj in adjectives) + ")$"

# Add color patterns to EntityRuler
ruler.add_patterns(color_patterns)

# Define keyword lists
brands = [
    "asus", "lenovo", "hp", "dell", "acer", "msi", "apple", "gigabyte", "huawei",
    "samsung", "xiaomi", "oppo", "vivo", "nokia", "realme", "oneplus", "sony",
    "google", "motorola", "honor", "zte", "infinix", "tecno", "alcatel", "blackberry",
    "meizu", "fairphone", "nothing", "redmi", "iqoo", "lava"
]

storage_keywords = [
    "festplatte", "ssd", "hdd", "speicher", "kapazität", "speicherkapazität", "datenträger"
]

# Initialize matcher
matcher = Matcher(nlp.vocab)

# BRAND_MODEL pattern
matcher.add("BRAND_MODEL", [[
    {"LOWER": {"IN": brands}},
    {"TEXT": {"REGEX": "^[a-zA-Z0-9\-]+$"}, "OP": "*"}
]])


# STORAGE patterns
matcher.add("STORAGE", [
    [
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["gb", "tb"]}, "OP": "?"},
        {"TEXT": {"REGEX": adjective_regex}, "OP": "?"},
        {"LOWER": {"IN": storage_keywords}}
    ],
    [
        {"TEXT": {"REGEX": adjective_regex}, "OP": "?"},
        {"LOWER": {"IN": storage_keywords}},
        {"LOWER": {"IN": ["gb", "tb"]}, "OP": "?"},
        {"LIKE_NUM": True}
    ],
    [
        {"TEXT": {"REGEX": adjective_regex}, "OP": "?"},
        {"LOWER": {"IN": storage_keywords}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["gb", "tb"]}, "OP": "?"}
    ],
    [
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["gb", "tb"]}},
        {"TEXT": {"REGEX": adjective_regex}, "OP": "?"},
        {"LOWER": {"IN": storage_keywords}}
    ],
    [
        {"LOWER": {"IN": storage_keywords}},
        {"TEXT": {"REGEX": adjective_regex}, "OP": "?"},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["gb", "tb"]}}
    ]
])

# Attribute extraction
def extract_attributes(text):
    text = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", text)  # fix "125TB" → "125 TB"
    doc = nlp(text)
    attributes = {}
    for ent in doc.ents:
        if ent.label_ in ["COLOR"]:
            attributes[ent.label_] = ent.text

    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        if label not in attributes:
            attributes[label] = span
    return attributes

# Find match positions in text
def find_all_matches(text, value):
    return [(m.start(), m.end()) for m in re.finditer(re.escape(value), text, flags=re.IGNORECASE)]

# Convert to spaCy training format
def create_ner_training_data(df, extracted_results):
    ner_data = []
    for idx, row in df.iterrows():
        raw_text = " ".join(str(val) for val in row.values if pd.notnull(val))
        text = clean_text(raw_text)
        annotations = []
        attributes = extracted_results[idx]
        for label, value in attributes.items():
            for part in value.split(" | "):
                matches = find_all_matches(text, part)
                if matches:
                    start, end = matches[0]
                    annotations.append((start, end, label))
        ner_data.append((text, {"entities": annotations}))
    return ner_data

# Run matcher on full dataset
results = []
for _, row in df.iterrows():
    raw_text = " ".join(str(val) for val in row.values if pd.notnull(val))
    desc = clean_text(raw_text)
    extracted = extract_attributes(desc)
    results.append(extracted)

# Generate and split training data
full_data = create_ner_training_data(df, results)
train_data, dev_data = train_test_split(full_data, test_size=0.2, random_state=42)

# Save .pkl
with open("ner_training_data.pkl", "wb") as f:
    pickle.dump(full_data, f)
print("Saved ner_training_data.pkl")

# Save to .spacy format
def save_spacy_file(data, filename):
    doc_bin = DocBin()
    for text, annot in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = filter_spans(ents)
        doc_bin.add(doc)
    doc_bin.to_disk(filename)
    print(f"Saved {filename}")

save_spacy_file(train_data, "data/train_data.spacy")
save_spacy_file(dev_data, "data/dev_data.spacy")

import pickle
with open("data/matcher.pkl", "wb") as f:
    pickle.dump(matcher, f)
print("Saved matcher.pkl")

