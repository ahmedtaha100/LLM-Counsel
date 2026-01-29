import argparse
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

SAMPLE_DATA = [
    ("What is 2+2?", 0),
    ("Hello", 0),
    ("What time is it?", 0),
    ("Define photosynthesis", 0),
    ("Translate hello to Spanish", 0),
    ("Explain the difference between TCP and UDP protocols", 1),
    ("Write a function to reverse a string in Python", 1),
    ("Compare the French and American revolutions", 1),
    ("What are the pros and cons of microservices?", 1),
    ("How does garbage collection work in Java?", 1),
    ("Design a distributed cache system for a high-traffic website", 2),
    ("Analyze the economic implications of universal basic income", 2),
    ("Write a recursive descent parser for arithmetic expressions", 2),
    ("Explain quantum entanglement and its applications in computing", 2),
    ("Create a machine learning pipeline for fraud detection with feature engineering", 2),
    ("Summarize this text", 0),
    ("Fix this bug", 1),
    ("Implement a REST API", 1),
    ("Architect a real-time bidding system", 2),
    ("Prove P != NP", 2),
]


def train(output_path: Path, encoder_name: str = "all-MiniLM-L6-v2") -> None:
    encoder = SentenceTransformer(encoder_name)

    texts = [item[0] for item in SAMPLE_DATA]
    labels = [item[1] for item in SAMPLE_DATA]

    embeddings = encoder.encode(texts)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    scores = cross_val_score(clf, embeddings, labels, cv=3)
    print(f"Cross-validation accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

    clf.fit(embeddings, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/complexity_model.pkl"))
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    train(args.output, args.encoder)
