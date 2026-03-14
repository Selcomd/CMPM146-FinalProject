# Trains the LSTM on the pseudo-labels we generated.
# Give it one or more audio files (or a directory), it builds the timing grid,
# makes labels for all 5 difficulties, and trains the model to reproduce
# those patterns from audio features alone. Saves the weights to stepLSTM_model.pth.
#
# Usage:
#   python3 trainModel.py                        (file picker, single file)
#   python3 trainModel.py path/to/song.mp3        (single file)
#   python3 trainModel.py song1.mp3 song2.mp3     (multiple files)
#   python3 trainModel.py path/to/music_folder/   (all audio in directory)

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import librosa

from timingGrid import TimingGrid
from lstmModel import StepLSTM
from dataGenerator import DIFFICULTY_SETTINGS, generate_pseudo_labels

MODEL_PATH = "stepLSTM_model.pth"

SEQ_LEN    = 64    # feed the LSTM 64 positions at a time (4 measures)
STRIDE     = 32    # 50% overlap between windows so we get more training samples
BATCH_SIZE = 32
EPOCHS     = 120
LR         = 1e-3
POS_WEIGHT = 4.0   # penalize missing a step 4x more than adding a wrong one


class StepDataset(Dataset):
    # Cuts the full song feature/label arrays into overlapping 64-step windows.
    # Each window becomes one training sample.

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        assert features.shape[0] == labels.shape[0]
        self.X, self.Y = [], []

        n = features.shape[0]
        for start in range(0, n - SEQ_LEN + 1, STRIDE):
            self.X.append(features[start : start + SEQ_LEN])
            self.Y.append(labels[start : start + SEQ_LEN])

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def _build_training_arrays(audio_path: str):
    # Load the audio, build the timing grid, then generate pseudo-labels for
    # all 5 difficulties. Stack them all together so the model sees the same
    # audio features paired with both sparse and dense arrow patterns.
    # This 5x data augmentation helps it learn a range of outputs.
    print(f"[train] Loading audio: {audio_path}")
    audio_data, sr = librosa.load(audio_path, sr=None, mono=True)

    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio_data, sr=sr, onset_envelope=onset_env
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    tempo_val = float(np.atleast_1d(tempo)[0])
    print(f"[train] BPM: {tempo_val:.1f}  |  Beats: {len(beat_times)}")

    grid = TimingGrid(audio_data, sr, tempo_val, beat_times).build_grid()
    print(f"[train] Grid size: {len(grid)} 16th-note positions")

    features = np.stack([cell["features"] for cell in grid])  # (N, 8)

    all_X, all_Y = [], []
    for diff in DIFFICULTY_SETTINGS:
        labels = generate_pseudo_labels(grid, diff)  # (N, 4)
        all_X.append(features)
        all_Y.append(labels)

    return np.vstack(all_X), np.vstack(all_Y)  # (N*5, 8) and (N*5, 4)


AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}


def _collect_audio_paths(paths: list[str]) -> list[str]:
    """Expand paths into a flat list of audio file paths.
    If a path is a directory, include all audio files inside it (non-recursive).
    """
    collected = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if not os.path.exists(p):
            print(f"[train] Warning: path does not exist, skipping: {p}")
            continue
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                collected.append(p)
            else:
                print(f"[train] Warning: not an audio file, skipping: {p}")
        else:
            for name in sorted(os.listdir(p)):
                ext = os.path.splitext(name)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    collected.append(os.path.join(p, name))
    return collected


def _build_training_arrays_multi(audio_paths: list[str]):
    """Build combined (features, labels) from multiple audio files."""
    if not audio_paths:
        raise ValueError("No audio paths provided")
    all_features = []
    all_labels = []
    for path in audio_paths:
        feat, lab = _build_training_arrays(path)
        all_features.append(feat)
        all_labels.append(lab)
    return np.vstack(all_features), np.vstack(all_labels)


def train_model(
    audio_path_or_paths,
    epochs: int = EPOCHS,
    save_path: str = MODEL_PATH,
) -> StepLSTM:
    """Train on one or more audio files. Pass a single path string or a list of paths."""
    if isinstance(audio_path_or_paths, (list, tuple)):
        paths = _collect_audio_paths(list(audio_path_or_paths))
        if not paths:
            raise ValueError("No valid audio files found in the given paths")
        print(f"[train] Multi-song training on {len(paths)} file(s)")
        features, labels = _build_training_arrays_multi(paths)
    else:
        paths = _collect_audio_paths([audio_path_or_paths])
        if not paths:
            raise ValueError("No valid audio file specified")
        features, labels = _build_training_arrays(paths[0])

    dataset = StepDataset(features, labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"[train] Training windows: {len(dataset)}")

    model     = StepLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss(reduction="none")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X, Y in loader:
            optimizer.zero_grad()
            pred = model(X)          # (batch, 64, 4)
            raw  = criterion(pred, Y)

            # Up-weight the loss whenever the true label is 1 (step present).
            # Without this, the model would just predict 0 everywhere and cheat.
            weight = torch.where(Y > 0.5, torch.full_like(Y, POS_WEIGHT), torch.ones_like(Y))
            loss = (raw * weight).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # keeps gradients stable
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0 or epoch == epochs:
            avg = epoch_loss / max(len(loader), 1)
            print(f"[train] Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[train] Model saved → {save_path}")
    model.eval()
    return model


def load_model(path: str = MODEL_PATH) -> StepLSTM:
    model = StepLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # One or more paths: files and/or directories
        paths = sys.argv[1:]
        if paths[0] == "--dry-run":
            # Only collect and print paths, no training (for testing path expansion)
            paths = paths[1:] if len(paths) > 1 else []
            if not paths:
                print("Usage: python3 trainModel.py --dry-run <path> [path ...]")
                sys.exit(1)
            collected = _collect_audio_paths(paths)
            print(f"Collected {len(collected)} audio file(s):")
            for p in collected:
                print(f"  {p}")
            sys.exit(0)
        train_model(paths)
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        audio_file = filedialog.askopenfilename(
            title="Select audio file for training",
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac"), ("All Files", "*.*")],
        )
        root.destroy()

        if not audio_file:
            print("No file selected.")
            sys.exit(1)

        train_model(audio_file)
