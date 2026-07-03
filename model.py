"""
model.py
Train a RandomForest fraud detection model for the Smart Contactless Hotel Experience demo.

Run:
    python dataset_generator.py
    python model.py
"""

from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data") / "hotel_dataset.csv"
MODEL_FILE = Path("model.pkl")
FEATURES = ["stay_days", "room_rate", "total_bill", "biometric_score"]
TARGET = "fraud_flag"


def train_model() -> RandomForestClassifier:
    """Load dataset, train model, save model.pkl, and return trained model."""
    if not DATA_FILE.exists():
        raise FileNotFoundError("Dataset not found. Run: python dataset_generator.py")

    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)

    print(f"Model trained and saved as: {MODEL_FILE}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


if __name__ == "__main__":
    train_model()
