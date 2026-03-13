# CMPM146-FinalProject
Automatically generates DDR-style StepMania beatmaps from any audio file.

## Team
Ethan Cao, Yahir Rico, Evangelene Stanley, Bolor Battulga

## Requirements
- Python 3.12
- run this in terminal: `pip install -r requirements.txt`

## How to run

Train the LSTM on a song:
```bash
python3 trainModel.py path/to/song.mp3
```
Or just run it without arguments and a file picker will open.

For just audio analysis:
```bash
python3 audioExtraction.py
```

## Files
- `audioExtraction.py` - Extracts BPM, beat times, and onsets from audio
- `timingGrid.py` - Snaps the song onto a 16th-note grid and pulls audio features for each slot
- `lstmModel.py` - The LSTM model that learns where to place arrows
- `dataGenerator.py` - Creates training labels using music rules since we have no real DDR charts
- `trainModel.py` - Trains the LSTM and saves the weights to `stepLSTM_model.pth`
