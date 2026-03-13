import torch
import torch.nn as nn

N_FEATURES = 8   # one feature vector per 16th-note position (see timingGrid.py)
N_ARROWS = 4     # Left, Down, Up, Right


class StepLSTM(nn.Module):
    # A bidirectional LSTM that reads the sequence of audio feature vectors
    # and predicts, for each 16th-note slot, how likely each arrow column is.
    # "Bidirectional" means it looks at context both before and after each
    # position — helpful for things like building up before a big drop.

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,    # 8 features coming in
            hidden_size=hidden_size,  # 128 hidden units per direction
            num_layers=num_layers,    # 2 stacked LSTM layers
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,       # runs forward and backward, then concatenates
        )

        # Small classifier that takes the LSTM output and squashes it
        # down to 4 probabilities (one per arrow column)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 because fwd + bwd are concatenated
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, N_ARROWS),
            nn.Sigmoid(),  # output is a probability in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape:   (batch, seq_len, 8)
        # out shape: (batch, seq_len, 4)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out)
