"""Train an SVM baseline on summary MFCC features."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.audio_features import collect_paths, mfcc_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    paths, y = collect_paths(args.data_dir)
    X = np.vstack([mfcc_summary(p) for p in paths])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Files: {len(paths)} | Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report (0=human, 1=AI-generated):")
    print(classification_report(y_test, predictions, target_names=["human", "ai_generated"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "svm_pipeline.joblib")
    print(f"Saved model to {output / 'svm_pipeline.joblib'}")

if __name__ == "__main__":
    main()
