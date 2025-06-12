# 🧠 German NER Product Info Extraction

This project is a full Named Entity Recognition (NER) pipeline for extracting structured product information (e.g. brand, storage, color) from messy German e-commerce texts. It includes rule-based + machine learning (hybrid) entity extraction, training, evaluation, logging, and a FastAPI service for serving predictions.

---

## 📁 Project Structure

```bash
ner_project/
├── data/                   # Cleaned and processed text data
│   ├── ds_ner_test_case.csv
│   ├── train_data.spacy
│   ├── dev_data.spacy
│   ├── matcher.pkl
│   ├── ner_training_data.pkl
├── output/                 # Trained model artifacts
├── logs/                   # Evaluation logs
├── plots/                  # Confusion matrix, entity distribution
├── main.py                 # Data cleaning, rule-based tagging, export to .spacy
├── train.py                # Model training with spaCy config
├── evaluate.py             # Evaluation with metrics + confusion matrix
├── apifast.py              # FastAPI hybrid service
├── test_api.py             # Unit tests for API
├── test_extract.py         # Unit tests for matcher-based extraction
├── distribution.py         # Visualize entity frequency in training data
├── config.cfg              # spaCy configuration file
├── Dockerfile              # Docker image setup
├── docker-compose.yml      # Docker orchestration for API
├── requirements.txt        # All dependencies
└── README.md               # 📖 Project usage guide (you're reading this!)
```

---

## 🚀 Quickstart (Development)

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Clean + Prepare Data

```bash
python main.py
```

Generates:

* `ner_training_data.pkl`
* `train_data.spacy`, `dev_data.spacy`
* `matcher.pkl`

### 3. Visualize Distribution

```bash
python distribution.py
```

Creates bar plots of frequency of labels (e.g. top brands, storage, colors).
This section has a deeper look at the data used for train. It can help us to see how labels (brands, storage, colors) were distributed and which label is more and less frequent. It is useful also not to allow the model be overfitted.

### 4. Train Model

```bash
python train.py
```

Outputs model to `output/model-best`.

**Training Configuration:**

* Language: `de` (German)
* Model: spaCy's `ner` component with default architecture
* Batch size: `128`
* Dropout: `0.1`
* Max epochs: `30`
* Eval frequency: `50`
* Config file used: `config.cfg`

### 5. Evaluate Model

```bash
python evaluate.py
```

Logs:

* Precision, Recall, F1 Score
* Confusion Matrix (in `plots/confusion_matrix.png`)
* Logs in `logs/evaluation.log`

---

## 🧪 Unit Testing

Run API and extraction tests:

```bash
python test_api.py
python test_extract.py
```

---

## 📦 Run FastAPI Server

```bash
uvicorn apifast:app --reload
```

Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example request:

```json
{
  "text": "acer 250-G8 Laptop mit 1 TB superschnell Speicherkapazität in silberene"
}
```

---

## 🐳 Docker Deployment

### Build the image:

```bash
docker build -t ner-app .
```

### Run container:

```bash
docker run -p 8000:8000 ner-app
```

Or use docker-compose:

```bash
docker-compose up --build
```



## ✅ Deliverable Features

* [x] Entity cleaning, tagging, saving
* [x] Hybrid matcher + ML model
* [x] Balanced train/dev splits
* [x] Visualizations of entity distribution
* [x] Evaluation with metrics + confusion matrix
* [x] Logging to `logs/`
* [x] FastAPI service
* [x] Dockerized
* [x] Unit tests (API + Extractor)

---

## 📌 Notes for Reviewers

* You can run **everything from scratch** using only the code in this repo.
* Output folders will be auto-created where needed.
* The API will return predictions even if only rule-based matcher is available (fallback logic).
* Model configuration and training metadata can be found in `config.cfg`

Feel free to explore and test!

---

Made with ❤️ and Python.

