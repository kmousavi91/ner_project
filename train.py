import subprocess

subprocess.run([
    "python", "-m", "spacy", "train", "config.cfg",
    "--output", "output",
    "--paths.train", "data/train_data.spacy",
    "--paths.dev", "data/dev_data.spacy",
    "--initialize.vectors", "de_core_news_md"
])
