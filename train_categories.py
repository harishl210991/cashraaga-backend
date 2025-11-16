# train_categories.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Categories you want CashRaaga to predict
TARGET_CATEGORIES = [
    "Salary",
    "Rent",
    "EMI",
    "Food & Dining",
    "Groceries",
    "Shopping",
    "Transfers",
    "UPI Payment",
    "Bank Transfer",
    "Income",
    "Fuel & Transport",
    "Mobile & Internet",
    "Bills & Utilities",
    "Others",
]

def main():
    # 1. Load your labelled data
    df = pd.read_csv("training_transactions.csv")

    # Keep only rows with non-empty description & category
    df = df.dropna(subset=["description", "category"])
    df["description"] = df["description"].astype(str)
    df["category"] = df["category"].astype(str)

    # Optionally filter to known categories
    df = df[df["category"].isin(TARGET_CATEGORIES)]

    X = df["description"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    # 2. Pipeline: TF-IDF + Logistic Regression
    model = Pipeline(
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
                    max_iter=300,
                    n_jobs=-1,
                    multi_class="auto",
                ),
            ),
        ]
    )

    # 3. Train
    model.fit(X_train, y_train)

    # 4. Simple accuracy print (optional)
    acc = model.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")
    print(f"Classes learned: {list(model.classes_)}")

    # 5. Save to disk – analysis.py will load this
    joblib.dump(model, "category_model.joblib")
    print("Saved model to category_model.joblib")

if __name__ == "__main__":
    main()
