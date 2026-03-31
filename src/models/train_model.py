
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.build_features import generate_tfidf
from src.data.load_data import load_processed_data

from src.evaluation.evaluate import evaluate_classification
from src.evaluation.validation import (
    cross_validate_model,
    check_class_balance
)
from src.evaluation.interpretability import show_feature_importance

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"

def prepare_target(df: pd.DataFrame, mode: str = "classification") -> pd.Series:
    """
    Prepare target variable.
    classification: binary sentiment
    regression/multiclass: rating
    """
    if mode == "classification":
        return (df["rating"] >= 4).astype(int)  # positiv vs negativ
    else:
        return df["rating"]

def train_model(model_type: str = "xgboost", mode: str = "classification"):
    # Load data
    df = load_processed_data()

    # Drop missing
    df = df.dropna(subset=["review_text_clean", "rating"])

    # Target
    y = prepare_target(df, mode)

    # Features
    preprocessor, df = generate_tfidf(df)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    # Model selection
    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss"
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:
        raise ValueError("Unknown model type")

    # Full Pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nModel: {model_type}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    #erweiterungen für evaluation, validation und interpretability,
    #todo anappassen für regression/multiclass falls nötig
    # Evaluation
    evaluate_classification(y_test, y_pred)

    # Validation
    check_class_balance(y)
    cross_validate_model(pipeline, df, y)

    # Interpretability
    show_feature_importance(pipeline)



    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return pipeline


if __name__ == "__main__":
    train_model(model_type="xgboost")