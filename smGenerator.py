# Arrow columns:  0=Left  1=Down  2=Up  3=Right
#
# Usage:
#   python3 smGenerator.py path/to/song.mp3
#   python3 smGenerator.py song.mp3 -d Medium,Hard

import argparse
import os
import sys
import numpy as np
import torch
import librosa

from timingGrid import TimingGrid
from lstmModel import StepLSTM
from trainModel import load_model

# One threshold per difficulty
THRESHOLDS = {
    "Beginner":  0.55,
    "Easy":      0.50,
    "Medium":    0.45,
    "Hard":      0.38,
    "Challenge": 0.30,
}

# StepMania numeric difficulty ratings
DIFFICULTY_RATINGS = {
    "Beginner":  1,
    "Easy":      3,
    "Medium":    5,
    "Hard":      7,
    "Challenge": 9,
}

SUBDIVISIONS_PER_MEASURE = 16


def predict_arrows(model: StepLSTM, grid: list, seq_len: int = 64) -> np.ndarray:
    n = len(grid)
    features = np.stack([cell["features"] for cell in grid])

    prob_sum   = np.zeros((n, 4), dtype=np.float32)
    prob_count = np.zeros((n, 1), dtype=np.float32)

    stride = seq_len // 2

    model.eval()
    with torch.no_grad():
        for start in range(0, n, stride):
            end    = min(start + seq_len, n)
            window = features[start:end]

            if len(window) < seq_len:
                pad    = np.zeros((seq_len - len(window), window.shape[1]), dtype=np.float32)
                window = np.vstack([window, pad])

            x    = torch.from_numpy(window).unsqueeze(0)
            pred = model(x).squeeze(0).numpy()

            actual_len = end - start
            prob_sum[start:end]   += pred[:actual_len]
            prob_count[start:end] += 1

    # Average overlapping predictions
    probs = prob_sum / np.maximum(prob_count, 1)
    return probs


def probs_to_arrows(probs: np.ndarray, threshold: float, difficulty: str = "Medium") -> np.ndarray:
    arrows = np.zeros((len(probs), 4), dtype=np.float32)

    col_counts = np.zeros(4, dtype=np.int32)
    recent_cols = []
    recent_pairs = []

    last_note_idx = -999
    last_single = None
    last_pair = None

    if difficulty == "Beginner":
        allowed_pairs = set()
        min_gap = 3          # minimum distance between notes
        target_gap = 5       # preferred spacing
        max_gap = 7          # force a note if we wait this long
        recent_window = 4
    elif difficulty == "Easy":
        allowed_pairs = {(0, 3)}
        min_gap = 2
        target_gap = 4
        max_gap = 6
        recent_window = 6
    elif difficulty == "Medium":
        allowed_pairs = {(0, 3), (0, 2), (1, 3)}
        min_gap = 1
        target_gap = 3
        max_gap = 5
        recent_window = 8
    else:
        allowed_pairs = {(0, 3), (0, 2), (1, 3)}
        min_gap = 1
        target_gap = 2
        max_gap = 4
        recent_window = 8

    center_cols = {1, 2}
    side_cols = {0, 3}

    for i, p in enumerate(probs):
        time_since_last = i - last_note_idx

        # Normal candidates
        candidate_cols = [c for c in range(4) if p[c] >= threshold]

        # If we're getting close to too much dead space, relax a little
        if not candidate_cols and time_since_last >= target_gap:
            relaxed_threshold = threshold * 0.92
            candidate_cols = [c for c in range(4) if p[c] >= relaxed_threshold]

        # If we've waited too long, force the best available column
        if not candidate_cols and time_since_last >= max_gap:
            best_any = int(np.argmax(p))
            candidate_cols = [best_any]

        if not candidate_cols or time_since_last < min_gap:
            continue

        avg_count = np.mean(col_counts) if np.sum(col_counts) > 0 else 0.0

        recent_counts = np.zeros(4, dtype=np.int32)
        for c in recent_cols[-recent_window:]:
            recent_counts[c] += 1

        side_total = col_counts[0] + col_counts[3]
        center_total = col_counts[1] + col_counts[2]

        scored = []
        for c in candidate_cols:
            score = float(p[c])

            col_mean = np.mean(col_counts) if np.sum(col_counts) > 0 else 0.0
            col_diff = col_counts[c] - col_mean
            score -= 0.10 * col_diff

            side_total = col_counts[0] + col_counts[3]
            center_total = col_counts[1] + col_counts[2]
            if c in side_cols and side_total + 2 < center_total:
                score += 0.05

            if c == last_single:
                score -= 0.18

            score -= 0.08 * recent_counts[c]

            scored.append((score, c))

        scored.sort(reverse=True)
        best_col = scored[0][1]

        placed_jump = False
        if (
                allowed_pairs
                and len(candidate_cols) >= 2
                and max(min_gap, 3) <= time_since_last < max_gap  # don't force jumps in dead-space recovery
        ):
            second_choices = [c for _, c in scored[1:]]

            for second_col in second_choices:
                pair = tuple(sorted((best_col, second_col)))

                if pair not in allowed_pairs:
                    continue
                if pair == last_pair or pair in recent_pairs[-3:]:
                    continue
                if pair == (1, 2):
                    continue
                if not ({best_col, second_col} & side_cols):
                    continue

                if p[best_col] >= threshold and p[second_col] >= threshold * 0.98:
                    arrows[i, best_col] = 1.0
                    arrows[i, second_col] = 1.0
                    col_counts[best_col] += 1
                    col_counts[second_col] += 1
                    recent_cols.extend([best_col, second_col])
                    recent_pairs.append(pair)
                    last_pair = pair
                    last_single = None
                    last_note_idx = i
                    placed_jump = True
                    break

        if placed_jump:
            continue

        arrows[i, best_col] = 1.0
        col_counts[best_col] += 1
        recent_cols.append(best_col)
        last_single = best_col
        last_pair = None
        last_note_idx = i

    return arrows


def format_measure(arrows_in_measure: np.ndarray) -> str:
    rows = []
    for row in arrows_in_measure:
        rows.append("".join("1" if v > 0.5 else "0" for v in row))
    return "\n".join(rows)

def build_sm_content(
    song_name: str,
    audio_filename: str,
    tempo: float,
    offset: float,
    all_arrows: dict,
) -> str:
    """
    Assembles the full text content of a .sm file.
    all_arrows is a dict mapping difficulty name -> (N, 4) binary array.
    """
    lines = []

    #Global metadata
    lines += [
        f"#TITLE:{song_name};",
        f"#SUBTITLE:;",
        f"#ARTIST:Unknown;",
        f"#TITLETRANSLIT:;",
        f"#SUBTITLETRANSLIT:;",
        f"#ARTISTTRANSLIT:;",
        f"#GENRE:;",
        f"#CREDIT:Auto-generated by smGenerator.py;",
        f"#BANNER:;",
        f"#BACKGROUND:;",
        f"#LYRICSPATH:;",
        f"#CDTITLE:;",
        f"#MUSIC:{audio_filename};",
        f"#OFFSET:{offset:.3f};",
        f"#SAMPLESTART:0.000;",
        f"#SAMPLELENGTH:10.000;",
        f"#SELECTABLE:YES;",
        f"#BPMS:0.000={tempo:.3f};",
        f"#STOPS:;",
        "",
    ]

    for diff, arrows in all_arrows.items():
        n         = len(arrows)
        rating    = DIFFICULTY_RATINGS.get(diff, 5)

        remainder = n % SUBDIVISIONS_PER_MEASURE
        if remainder != 0:
            pad    = np.zeros((SUBDIVISIONS_PER_MEASURE - remainder, 4), dtype=np.float32)
            arrows = np.vstack([arrows, pad])

        n_measures = len(arrows) // SUBDIVISIONS_PER_MEASURE

        lines += [
            "#NOTES:",
            "     dance-single:",
            "     :",
            f"     {diff}:",
            f"     {rating}:",
            "     0.000,0.000,0.000,0.000,0.000:",
        ]

        for m in range(n_measures):
            chunk        = arrows[m * SUBDIVISIONS_PER_MEASURE : (m + 1) * SUBDIVISIONS_PER_MEASURE]
            measure_str  = format_measure(chunk)
            separator    = ";" if m == n_measures - 1 else ","
            lines.append(measure_str)
            lines.append(separator)

        lines.append("")

    return "\n".join(lines)


def generate_sm(
    audio_path: str,
    model: StepLSTM,
    output_dir: str = ".",
    difficulties: list = None,
) -> str:
    if difficulties is None:
        difficulties = list(THRESHOLDS.keys())

    print(f"\n[smGenerator] Loading audio: {audio_path}")
    audio_data, sr = librosa.load(audio_path, sr=None, mono=True)

    onset_env          = librosa.onset.onset_strength(y=audio_data, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr, onset_envelope=onset_env)
    beat_times         = librosa.frames_to_time(beat_frames, sr=sr)
    tempo_val          = float(np.atleast_1d(tempo)[0])
    duration           = librosa.get_duration(y=audio_data, sr=sr)
    offset             = float(beat_times[0]) if len(beat_times) > 0 else 0.0

    print(f"[smGenerator] BPM={tempo_val:.1f}  Duration={duration:.1f}s  Offset={offset:.3f}s")

    grid  = TimingGrid(audio_data, sr, tempo_val, beat_times).build_grid()
    probs = predict_arrows(model, grid)
    print(f"[smGenerator] Grid positions: {len(grid)}")

    all_arrows = {}
    for diff in difficulties:
        threshold         = THRESHOLDS[diff]
        arrows            = probs_to_arrows(probs, threshold, diff)
        density           = arrows.any(axis=1).mean()
        all_arrows[diff]  = arrows
        print(f"[smGenerator] {diff:10s}  threshold={threshold}  step_density={density:.2f}")

    song_name      = os.path.splitext(os.path.basename(audio_path))[0]
    audio_filename = os.path.basename(audio_path)
    sm_content     = build_sm_content(song_name, audio_filename, tempo_val, offset, all_arrows)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{song_name}.sm")
    with open(out_path, "w") as f:
        f.write(sm_content)

    print(f"[smGenerator] Saved → {out_path}")
    return out_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate StepMania .sm beatmaps from audio using the trained LSTM."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to audio file (.mp3, .wav, etc.). Omit to use file picker.",
    )
    parser.add_argument(
        "-d", "--difficulties",
        default=None,
        help="Comma-separated difficulties to generate, e.g. Medium,Hard. Default: all 5.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.audio:
        audio_file = args.audio
        if not os.path.isfile(audio_file):
            print(f"Error: file not found: {audio_file}")
            sys.exit(1)
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        audio_file = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac"), ("All Files", "*.*")],
        )
        root.destroy()

        if not audio_file:
            print("No file selected.")
            sys.exit(1)

    difficulties = None
    if args.difficulties:
        difficulties = [s.strip() for s in args.difficulties.split(",")]
        valid = set(THRESHOLDS.keys())
        invalid = [d for d in difficulties if d not in valid]
        if invalid:
            print(f"Error: invalid difficulty names: {invalid}")
            print(f"Valid: {', '.join(sorted(valid))}")
            sys.exit(1)

    model = load_model()
    out_path = generate_sm(
        audio_file,
        model,
        output_dir="./output",
        difficulties=difficulties,
    )
    print(f"\nDone! StepMania file saved to: {out_path}")