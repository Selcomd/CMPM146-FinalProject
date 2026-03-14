# CMPM146-FinalProject
Automatically generates DDR-style StepMania beatmaps from any audio file.

## Team
Ethan Cao, Yahir Rico, Evangelene Stanley, Bolor Battulga

## Requirements
- Python 3.12
- run this in terminal: `pip install -r requirements.txt`

## How to run

**Train** the LSTM on one or more songs:
```bash
python3 trainModel.py path/to/song.mp3
python3 trainModel.py song1.mp3 song2.mp3 song3.wav
python3 trainModel.py path/to/music_folder/
```
Or run without arguments and a file picker will open (single file). A directory path loads all supported audio files (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac`) from that folder.

**Generate** a StepMania chart from audio (requires a trained model):
```bash
python3 smGenerator.py path/to/song.mp3
python3 smGenerator.py song.mp3 -d Medium,Hard
```
- `-d` / `--difficulties`: Comma-separated list (e.g. `Medium,Hard`). Omit for all 5 difficulties

Output: `./output/{songname}.sm` — load the `.sm` file and audio in StepMania to play.

**Audio analysis only:**
```bash
python3 audioExtraction.py
```

## Files
- `audioExtraction.py` - Extracts BPM, beat times, and onsets from audio
- `timingGrid.py` - Snaps the song onto a 16th-note grid and pulls audio features for each slot
- `lstmModel.py` - The LSTM model that learns where to place arrows
- `dataGenerator.py` - Creates training labels using music rules since we have no real DDR charts
- `trainModel.py` - Trains the LSTM and saves the weights to `stepLSTM_model.pth`
- `smGenerator.py` - Loads audio, runs the model, and exports a playable StepMania `.sm` file
