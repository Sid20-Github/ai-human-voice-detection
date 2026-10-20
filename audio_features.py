"""Audio loading and MFCC feature extraction shared by training/prediction scripts."""
from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 16_000
N_MFCC = 40
MAX_FRAMES = 160

def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at a consistent sample rate."""
    signal, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"No audio samples found in: {path}")
    return signal.astype(np.float32)

def mfcc_sequence(path: str | Path, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Return a (max_frames, N_MFCC) MFCC sequence, padded/truncated in time."""
    signal = load_audio(path)
    features = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if len(features) < max_frames:
        pad = np.zeros((max_frames - len(features), N_MFCC), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[:max_frames]
    return features.astype(np.float32)

def mfcc_summary(path: str | Path) -> np.ndarray:
    """Return mean and standard deviation of MFCCs, suitable for a baseline SVM."""
    signal = load_audio(path)
    features = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.concatenate([features.mean(axis=1), features.std(axis=1)]).astype(np.float32)

def collect_paths(data_dir: str | Path):
    """Collect labelled audio paths from data_dir/{human,ai_generated}."""
    root = Path(data_dir)
    classes = [("human", 0), ("ai_generated", 1)]
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    paths, labels = [], []
    for folder, label in classes:
        class_dir = root / folder
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing {class_dir}. Create both data/human and data/ai_generated folders."
            )
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                paths.append(path)
                labels.append(label)
    if len(set(labels)) < 2:
        raise ValueError(
            "Need audio files in both data/human/ and data/ai_generated/ before training."
        )
    return paths, np.asarray(labels, dtype=np.int64)
