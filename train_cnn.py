"""Train a small 1D CNN on fixed-length MFCC sequences."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.audio_features import collect_paths, mfcc_sequence, N_MFCC, MAX_FRAMES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    paths, y = collect_paths(args.data_dir)
    X = np.stack([mfcc_sequence(p) for p in paths])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Normalise using training-set statistics only.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_FRAMES, N_MFCC)),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train, y_train, validation_split=0.2, epochs=args.epochs,
        batch_size=args.batch_size, callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ], verbose=1
    )
    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report (0=human, 1=AI-generated):")
    print(classification_report(y_test, predictions, target_names=["human", "ai_generated"], zero_division=0))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "voice_cnn.keras")
    np.savez(output / "cnn_normalization.npz", mean=mean, std=std)
    print(f"Saved model to {output / 'voice_cnn.keras'}")

if __name__ == "__main__":
    main()
