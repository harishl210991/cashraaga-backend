# train_categories.py
"""
Train the transaction category classifier for CashRaaga.

Usage:
    python train_categories.py --input training_transactions.csv \
                               --output category_model.joblib

Expected CSV columns:
    - description : transaction narration (string)
    - category    : label (e.g. Salary, Rent, EMI, Groceries, etc.)

Important:
    Make sure to include HSBC SALARY-like rows and label them as "Salary"
    so the model learns those patterns even without the word "salary".
"""

import argparse
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


DEFAULT_OUTPUT = "category_model.joblib"


def train_category_model(input_csv: str, output_path: str = DEFAULT_OUTPUT) -> None:
    # Load data
    df = pd.read_csv(input_csv)
    if "description" not in df.columns or "category" not in df.columns:
        raise ValueError(
            "Input CSV must contain columns: 'description' and 'category'. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["description", "category"]).copy()
    df["description"] = df["description"].astype(str)
    df["category"] = df["category"].astype(str)

    print(f"[train_categories] Loaded {len(df)} rows from {input_csv}")
    print("[train_categories] Category distribution:")
    print(df["category"].value_counts())

    X = df["description"]
    y = df["category"]

    # If only one class exists, we cannot train a proper classifier
    if y.nunique() < 2:
        raise ValueError("Need at least 2 different categories to train a model.")

    stratify_y = y if y.nunique() > 1 else None

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
                    min_df=2,
                    max_features=20000,
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

    print("[train_categories] Training model...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train_categories] Validation accuracy: {acc:.3f}")
    print("[train_categories] Classification report:")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(pipe, output_path)
    print(f"[train_categories] Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train category classifier model.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV with 'description' and 'category' columns.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for joblib model (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()
    train_category_model(args.input, args.output)


if __name__ == "__main__":
    main()
