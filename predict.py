"""Predict whether one audio file resembles the human or AI class."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from src.audio_features import mfcc_summary, mfcc_sequence

LABELS = {0: "human", 1: "AI-generated"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to an audio clip")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--model-type", choices=["svm", "cnn"], required=True)
    args = parser.parse_args()

    if args.model_type == "svm":
        model = joblib.load(args.model)
        features = mfcc_summary(args.audio).reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        prediction = int(model.predict(features)[0])
        confidence = float(probabilities[prediction])
    else:
        model_path = Path(args.model)
        norm_path = model_path.parent / "cnn_normalization.npz"
        if not norm_path.exists():
            raise FileNotFoundError(f"Expected normalisation file at {norm_path}")
        norm = np.load(norm_path)
        features = mfcc_sequence(args.audio)[None, ...]
        features = (features - norm["mean"]) / (norm["std"] + 1e-6)
        model = tf.keras.models.load_model(model_path)
        confidence = float(model.predict(features, verbose=0)[0, 0])
        prediction = int(confidence >= 0.5)
        confidence = confidence if prediction == 1 else 1.0 - confidence

    print(f"Prediction: {LABELS[prediction]}")
    print(f"Model score for predicted class: {confidence:.3f}")
    print("Note: this is an experimental model output, not proof of audio origin.")

if __name__ == "__main__":
    main()
