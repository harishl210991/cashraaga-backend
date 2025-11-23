# train_bank_model.py
"""
Train the bank template detection model for CashRaaga.

Usage:
    python train_bank_model.py --input bank_training_samples.csv \
                               --output bank_template_model.joblib

Expected CSV columns:
    - text : concatenated header + few rows from a statement
    - bank : bank label (e.g. ICICI, HDFC, HSBC, SBI, GENERIC)
"""

import argparse
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


DEFAULT_OUTPUT = "bank_template_model.joblib"


def train_bank_model(input_csv: str, output_path: str = DEFAULT_OUTPUT) -> None:
    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "bank" not in df.columns:
        raise ValueError(
            "Input CSV must contain columns: 'text' and 'bank'. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "bank"]).copy()
    df["text"] = df["text"].astype(str)
    df["bank"] = df["bank"].astype(str).str.upper()

    print(f"[train_bank_model] Loaded {len(df)} rows from {input_csv}")
    print("[train_bank_model] Bank distribution:")
    print(df["bank"].value_counts())

    X = df["text"]
    y = df["bank"]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 different banks to train a model.")
# Use stratify only if every class has ≥2 samples
value_counts = y.value_counts()
if (value_counts >= 2).all():
    stratify_y = y
else:
    stratify_y = None  # disable stratification

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_y,
)
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    n_jobs=-1,
                    multi_class="auto",
                ),
            ),
        ]
    )

    print("[train_bank_model] Training model...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train_bank_model] Validation accuracy: {acc:.3f}")
    print("[train_bank_model] Classification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipe, output_path)
    print(f"[train_bank_model] Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train bank template detection model.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV with 'text' and 'bank' columns.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for joblib model (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()
    train_bank_model(args.input, args.output)


if __name__ == "__main__":
    main()
