import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.load_data import load_processed_data
from src.features.build_features import generate_tfidf


OUTPUT_PATH = Path("models/model_comparison.csv")


def prepare_target(df, mode="classification"):
    if mode == "classification":
        return (df["rating"] >= 4).astype(int)
    else:
        return df["rating"]


def get_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    }


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Cross Validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_weighted")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "cv_mean_f1": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4)
    }


def compare_models():
    df = load_processed_data()
    df = df.dropna(subset=["review_text_clean", "rating"])

    y = prepare_target(df)
    preprocessor, df = generate_tfidf(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    results = []

    models = get_models()

    for name, model in models.items():
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

        # Save each model separately
        joblib.dump(pipeline, f"models/{name.replace(' ', '_')}.joblib")

    results_df = pd.DataFrame(results)

    # Sort by best model
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    #add run
    results_df["timestamp"] = datetime.now()

    # Save
    results_df.to_csv(OUTPUT_PATH,  mode="a", header=False, index=False)

    print("\nModel comparison saved to:", OUTPUT_PATH)
    print(results_df)

    return results_df


if __name__ == "__main__":
    compare_models()