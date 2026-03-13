import numpy as np
import librosa

HOP_LENGTH = 512
N_FEATURES = 8
SUBDIVISIONS = 4  # 4 subdivisions per beat = 16th note resolution


class TimingGrid:
    # Takes the raw audio + beat info from Week 1 and snaps everything
    # onto a strict 16th-note clock. Each slot on that clock gets a
    # little bundle of audio features the LSTM can actually learn from.

    def __init__(self, audio_data, sample_rate, tempo, beat_times, subdivisions=SUBDIVISIONS):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.tempo = float(np.atleast_1d(tempo)[0])
        self.beat_times = np.asarray(beat_times)
        self.subdivisions = subdivisions
        self.hop_length = HOP_LENGTH

    def _extract_features(self):
        # Pull four different audio signals we'll use as features.
        # All of them get normalized to [0, 1] so the LSTM isn't
        # thrown off by songs that are louder or quieter.
        y, sr, hop = self.audio_data, self.sample_rate, self.hop_length

        # How "eventful" each moment is — spikes on drum hits, strums, etc.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

        # "Brightness" of the sound — high = more treble, low = more bass
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

        # Raw loudness / energy at each frame
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]

        # Positive spectral flux — captures sudden energy jumps across
        # the mel frequency bands (another way to catch onsets)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        flux = np.diff(mel_db, axis=1, prepend=mel_db[:, :1])
        flux = np.maximum(flux, 0).mean(axis=0)  # only keep increases, not decreases

        def safe_norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)  # +tiny value avoids divide-by-zero

        return (
            safe_norm(onset_env).astype(np.float32),
            safe_norm(spec_centroid).astype(np.float32),
            safe_norm(rms).astype(np.float32),
            safe_norm(flux).astype(np.float32),
        )

    def build_grid(self):
        # Walk through the song one 16th note at a time, starting from the
        # first detected beat. At each step, snap to the nearest audio frame
        # and pack 8 features into a vector for the LSTM.
        onset_env, spec_centroid, rms, flux = self._extract_features()
        n_frames = len(onset_env)

        beat_duration = 60.0 / self.tempo           # seconds per beat
        subdiv_duration = beat_duration / self.subdivisions  # seconds per 16th note

        song_duration = len(self.audio_data) / self.sample_rate
        start_time = self.beat_times[0] if len(self.beat_times) > 0 else 0.0

        grid = []
        t = start_time
        grid_idx = 0

        steps_per_measure = self.subdivisions * 4  # 16 slots per measure

        while t < song_duration - subdiv_duration * 0.5:
            # Find which librosa frame is closest to this 16th-note position
            frame = int(np.clip(
                librosa.time_to_frames(t, sr=self.sample_rate, hop_length=self.hop_length),
                0, n_frames - 1,
            ))

            # Figure out where we are musically
            subdiv_in_measure = grid_idx % steps_per_measure  # 0..15
            beat_in_measure   = subdiv_in_measure // self.subdivisions  # 0..3
            subdiv_in_beat    = subdiv_in_measure % self.subdivisions   # 0..3
            measure_idx       = grid_idx // steps_per_measure

            is_quarter  = 1.0 if subdiv_in_beat == 0 else 0.0   # are we on a beat?
            is_downbeat = 1.0 if subdiv_in_measure == 0 else 0.0  # beat 1 of a measure?

            # Encode position within the measure as sin/cos so that the
            # start and end of a measure feel "close" to each other
            phase = subdiv_in_measure / steps_per_measure  # 0.0 → 1.0

            features = np.array([
                onset_env[frame],
                spec_centroid[frame] if frame < len(spec_centroid) else 0.0,
                rms[frame] if frame < len(rms) else 0.0,
                flux[frame] if frame < len(flux) else 0.0,
                is_quarter,
                is_downbeat,
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
            ], dtype=np.float32)

            grid.append({
                "time": t,
                "grid_idx": grid_idx,
                "measure_idx": measure_idx,
                "subdiv_in_measure": subdiv_in_measure,
                "beat_in_measure": beat_in_measure,
                "subdiv_in_beat": subdiv_in_beat,
                "features": features,
                "onset_strength": float(onset_env[frame]),
            })

            t += subdiv_duration
            grid_idx += 1

        return grid
