# train_bank_model.py
"""
Train the bank template detection model for CashRaaga.

Usage:
    python train_bank_model.py --input bank_training_samples.csv \
                               --output bank_template_model.joblib

Expected CSV columns:
    - text : concatenated header + a few rows from a statement
    - bank : bank label (e.g. ICICI, HDFC, HSBC, HDFC, GENERIC)
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
    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
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
        raise ValueError(
            "Need at least 2 different bank labels to train a model. "
            f"Currently found: {y.unique().tolist()}"
        )

    # ------------------------------------------------------------------
    # 2. HANDLE SMALL CLASS COUNTS (FIX FOR YOUR ERROR)
    # ------------------------------------------------------------------
    value_counts = y.value_counts()
    print("[train_bank_model] Class counts:")
    print(value_counts)

    # We only use stratify if every class has at least 2 samples
    if (value_counts >= 2).all():
        stratify_y = y
        print("[train_bank_model] Using stratified train/test split.")
    else:
        stratify_y = None
        print(
            "[train_bank_model] Some classes have < 2 samples. "
            "Disabling stratified split (this is normal for tiny datasets)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_y,
    )

    # ------------------------------------------------------------------
    # 3. BUILD PIPELINE: TF-IDF + LOGISTIC REGRESSION
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. EVALUATE
    # ------------------------------------------------------------------
    if len(X_test) > 0:
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[train_bank_model] Validation accuracy: {acc:.3f}")
        print("[train_bank_model] Classification report:")
        print(classification_report(y_test, y_pred))
    else:
        print(
            "[train_bank_model] No test samples (very small dataset). "
            "Skipping evaluation."
        )

    # ------------------------------------------------------------------
    # 5. SAVE MODEL
    # ------------------------------------------------------------------
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
