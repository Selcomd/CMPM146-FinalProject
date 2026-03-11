import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import os


def analyze_song():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an MP3 file",
        filetypes=[("Audio Files", "*.mp3 *.wav *.ogg"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return

    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Basic info
        duration = librosa.get_duration(y=y, sr=sr)

        # Onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            onset_envelope=onset_env
        )
        tempo_value = float(np.atleast_1d(tempo)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            onset_envelope=onset_env
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Output results until we input them into the LTSM
        print("\n***AUDIO ANALYSIS PLACEHOLDER OUTPUT***")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Sample Rate: {sr}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Estimated BPM: {tempo_value:.2f}")
        print(f"Total Beats Detected: {len(beat_frames)}")
        print(f"Total Onsets Detected: {len(onset_frames)}")

        print("\nFirst 20 Beat Times (seconds):")
        print(np.round(beat_times[:20], 3))

        print("\nFirst 20 Onset Times (seconds):")
        print(np.round(onset_times[:20], 3))

        print("\nFirst 20 Beat Frames:")
        print(beat_frames[:20])

        print("\nFirst 20 Onset Frames:")
        print(onset_frames[:20])

        messagebox.showinfo("Done", "Audio analysis complete. Check the terminal output.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze audio:\n{e}")


if __name__ == "__main__":
    analyze_song()