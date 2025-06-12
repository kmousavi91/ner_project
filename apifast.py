from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.util import filter_spans
import uvicorn

app = FastAPI(title="NER Hybrid Extractor", version="1.0")

# Load trained model
# Load trained model
nlp = spacy.load("output/model-best")

# Add EntityRuler for color (just to ensure it's in place)
ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
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

# Add matcher after model
matcher = Matcher(nlp.vocab)

# Keywords for storage and brands
brands = ["asus", "lenovo", "hp", "dell", "acer", "msi", "apple", "gigabyte", "huawei"]
storage_keywords = [
    "festplatte", "ssd", "hdd", "speicher", "kapazität", "speicherkapazität", "datenträger"
]

adjectives = [
    "schnell",
    "hochgeschwindigkeit",
    "rasch",
    "ultraschnell",
    "leistungsstark",
    "effizient",
    "sauber",
    "klar",
    "strukturiert",
    "scharf",
    "poliert",
    "glatt"
]

adjective_regex = "(?i)^(schnell|ultraschnell|leistungsstark|hochwertig|...)"


# Rule for BRAND_MODEL like 'Asus 250-G8'
matcher.add("BRAND_MODEL", [[
    {"LOWER": {"IN": brands}},
    {"TEXT": {"REGEX": "^[a-zA-Z0-9\\-]+$"}, "OP": "?"},
    {"TEXT": {"REGEX": "^[a-zA-Z0-9\\-]+$"}, "OP": "?"}
]])

# Rule for STORAGE like '512GB SSD'
matcher.add("STORAGE", [
    # Original patterns
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
    # Enhanced patterns
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
# Input schema
class TextRequest(BaseModel):
    text: str

# API endpoint
@app.post("/extract")
async def extract_entities(request: TextRequest):
    text = request.text
    doc = nlp(text)
    matches = matcher(doc)

    new_ents = []
    for match_id, start, end in matches:
        span = Span(doc, start, end, label=nlp.vocab.strings[match_id])
        new_ents.append(span)

    doc.ents = filter_spans(list(doc.ents) + new_ents)

    return {
        "text": text,
        "entities": [{"label": ent.label_, "text": ent.text} for ent in doc.ents]
    }

# Auto-run Uvicorn when script is executed directly
if __name__ == "__main__":
    uvicorn.run("apifast:app", host="127.0.0.1", port=8000, reload=True)

