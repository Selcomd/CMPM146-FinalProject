# We don't have real DDR charts to train on, so we generate our own
# "pseudo-labels" using music rules. The idea is: place arrows where
# the audio is interesting (strong beats, loud hits), and follow DDR
# conventions like alternating feet and occasional jumps.
# These become the training targets for the LSTM.
#
# Arrow columns:  0=Left  1=Down  2=Up  3=Right

import numpy as np

# How dense each difficulty should be — fraction of 16th-note slots with a step
DIFFICULTY_SETTINGS = {
    "Beginner":  {"target_density": 0.05, "jump_boost": 0.92, "min_gap": 3},
    "Easy":      {"target_density": 0.08, "jump_boost": 0.85, "min_gap": 3},
    "Medium":    {"target_density": 0.20, "jump_boost": 0.75, "min_gap": 2},
    "Hard":      {"target_density": 0.40, "jump_boost": 0.65, "min_gap": 2},
    "Challenge": {"target_density": 0.60, "jump_boost": 0.55, "min_gap": 1},
}

# Cycle through these columns for single arrows so feet naturally alternate
_SINGLE_CYCLE = [0, 3, 1, 2, 3, 0, 2, 1]  # L R D U R L U D

# Arrow pairs used for jumps — chosen so they feel natural to hit together
_JUMP_PAIRS = [(0, 3), (1, 2), (0, 2), (1, 3)]


def generate_pseudo_labels(grid: list, difficulty: str = "Medium") -> np.ndarray:
    cfg            = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS["Medium"])
    target_density = cfg["target_density"]
    jump_boost     = cfg["jump_boost"]
    min_gap        = cfg["min_gap"]  # minimum 16th-note gap between steps

    n = len(grid)
    arrows = np.zeros((n, 4), dtype=np.float32)

    if n == 0:
        return arrows

    # Score each position — boost positions that land on a beat or downbeat
    # since DDR charts almost always prefer those spots
    scores = np.array([
        min(cell["onset_strength"]
            * (1.3 if cell["features"][4] > 0.5 else 1.0)   # on a quarter note beat
            * (1.2 if cell["features"][5] > 0.5 else 1.0),  # on beat 1 of a measure
            1.0)
        for cell in grid
    ], dtype=np.float32)

    # Set the threshold based on percentile so we always hit the target density,
    # regardless of how loud or quiet the song is
    pct = max(5.0, min(95.0, (1.0 - target_density) * 100.0))
    threshold = float(np.percentile(scores, pct))

    last_step_idx = -min_gap - 1
    last_col      = -1
    cycle_pos     = 0

    for i in range(n):
        if scores[i] < threshold:
            continue

        # Don't place steps too close together — that's physically impossible
        if (i - last_step_idx) < min_gap:
            continue

        # If the last 16 positions were already packed with steps, cool it down
        if i > 0:
            window_start  = max(0, i - 16)
            window        = arrows[window_start:i]
            local_density = window.any(axis=1).mean() if len(window) > 0 else 0.0
            if np.isnan(local_density):
                local_density = 0.0
            if local_density > target_density * 1.5:
                continue

        gap_since_last = i - last_step_idx

        if scores[i] >= jump_boost and gap_since_last >= 4:
            # Strong enough hit + enough space = place a jump (two arrows at once)
            pair = _JUMP_PAIRS[i % len(_JUMP_PAIRS)]
            arrows[i, pair[0]] = 1.0
            arrows[i, pair[1]] = 1.0
            last_col = pair[1]
        else:
            # Normal single arrow — cycle columns so feet alternate naturally
            candidates = [c for c in _SINGLE_CYCLE if c != last_col]
            chosen     = candidates[cycle_pos % len(candidates)]
            arrows[i, chosen] = 1.0
            last_col  = chosen
            cycle_pos += 1

        last_step_idx = i

    return arrows


def generate_all_difficulties(grid: list) -> dict:
    # Convenience wrapper — returns labels for all five difficulties at once
    return {diff: generate_pseudo_labels(grid, diff) for diff in DIFFICULTY_SETTINGS}
