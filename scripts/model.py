"""
model.py

PyTorch Deep Learning Model Architecture for Sport Activity Action Recognition.

Architecture: 2-Layer Bidirectional LSTM with Multi-Head Temporal Attention Pooling.

Input:  Tensor of shape (Batch, Seq_Len=30, Features=239)
        - 239 features per frame = 132 normalized coords + 8 joint angles + 99 velocity values
Output: Logits of shape (Batch, Num_Classes=3)
        - 3 sport actions: running, pushups, jumping_jacks

Why this architecture?
    - Bidirectional LSTM captures temporal motion patterns in both forward and backward
      directions, which is important because sport actions have distinct phases
      (e.g., pushup down→up or jumping jack spread→close).
    - Temporal Attention learns to focus on the most discriminative time steps
      (e.g., the peak of a jump, the bottom of a pushup) rather than treating
      all 30 frames equally.
    - LayerNorm + Dropout prevent overfitting on the relatively small dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    PyTorch Dataset wrapper for pose landmark sequence features and action labels.

    Converts numpy arrays to PyTorch tensors for use with DataLoader.

    Args:
        X: numpy array of shape (N, seq_len, features) — input feature sequences
        y: numpy array of shape (N,) — integer-encoded action labels
    """
    def __init__(self, X, y):
        # Convert to float32 tensors for model input
        self.X = torch.tensor(X, dtype=torch.float32)
        # Labels must be long (int64) for CrossEntropyLoss
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism that learns which time steps (frames) in a
    sequence are most important for action classification.

    Instead of using only the last LSTM hidden state or averaging all time steps,
    this module computes a learned weighted sum over the entire sequence.

    For example, in a pushup sequence, the attention may learn to focus on
    the lowest body position frame (most discriminative moment).

    Architecture:
        Linear(hidden_dim → 64) → Tanh → Linear(64 → 1) → Softmax over time

    Args:
        hidden_dim: size of the LSTM hidden state (hidden_dim * 2 for bidirectional)
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),   # Project hidden state to attention space
            nn.Tanh(),                    # Non-linear activation
            nn.Linear(64, 1)             # Scalar attention score per time step
        )

    def forward(self, lstm_out):
        """
        Args:
            lstm_out: shape (batch_size, seq_len, hidden_dim)

        Returns:
            context: shape (batch_size, hidden_dim) — attention-weighted summary vector
            attn_weights: shape (batch_size, seq_len, 1) — learned attention weights
        """
        # Compute attention score for each time step
        attn_weights = self.attn(lstm_out)              # (batch, seq_len, 1)
        # Softmax across time dimension → weights sum to 1.0
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum: multiply each LSTM output by its attention weight, then sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_dim)
        return context, attn_weights


class SportActionLSTM(nn.Module):
    """
    Main model: Multi-layer Bidirectional LSTM with Temporal Attention for
    Sport Activity Action Recognition.

    Data flow:
        Input (batch, 30, 239)
            ↓
        Bidirectional LSTM × 2 layers → (batch, 30, 256)   [128 forward + 128 backward]
            ↓
        Temporal Attention Pooling → (batch, 256)            [weighted sum over 30 frames]
            ↓
        LayerNorm → Dropout → FC(256→128) → GELU → Dropout → FC(128→3)
            ↓
        Output logits (batch, 3)

    Args:
        input_dim:  number of features per frame (default 239)
        hidden_dim: LSTM hidden size per direction (default 128, total 256 bidirectional)
        num_layers: number of stacked LSTM layers (default 2)
        num_classes: number of sport action classes (default 3)
        dropout:    dropout probability for regularization (default 0.3)
    """
    def __init__(self, input_dim: int = 239, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super(SportActionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Bidirectional LSTM ──
        # Processes the 30-frame sequence in both forward (t=0→29) and backward (t=29→0)
        # directions, capturing temporal context from both past and future frames.
        # Output dimension = hidden_dim × 2 = 256 (concatenated forward + backward)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,                              # Input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,    # Inter-layer dropout (only if >1 layer)
            bidirectional=True
        )

        # ── Temporal Attention ──
        # Learns to weight each of the 30 frames by importance
        self.attention = TemporalAttention(hidden_dim * 2)

        # ── Normalization & Regularization ──
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # Stabilizes training
        self.fc_dropout = nn.Dropout(dropout)            # Prevents overfitting

        # ── Classification Head ──
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Hidden projection layer
        self.act = nn.GELU()                         # Smooth activation (better than ReLU for small models)
        self.fc2 = nn.Linear(128, num_classes)       # Final classification layer (logits)

    def forward(self, x):
        """
        Forward pass: sequence → action class logits.

        Args:
            x: input tensor of shape (batch_size, seq_len=30, input_dim=239)

        Returns:
            logits: tensor of shape (batch_size, num_classes=3) — raw class scores
        """
        # Pass sequence through bidirectional LSTM
        lstm_out, _ = self.lstm(x)         # (batch, 30, 256)

        # Apply temporal attention to get a single summary vector per sequence
        context, _ = self.attention(lstm_out)  # (batch, 256)

        # Classification head with normalization and regularization
        norm_out = self.layer_norm(context)
        out = self.fc_dropout(norm_out)
        out = self.act(self.fc1(out))      # (batch, 128)
        out = self.fc_dropout(out)
        logits = self.fc2(out)             # (batch, 3)
        return logits


def get_model(input_dim: int = 239, num_classes: int = 3, hidden_dim: int = 128) -> nn.Module:
    """
    Factory function to create a SportActionLSTM model instance.

    Args:
        input_dim:   feature dimension per frame (default 239)
        num_classes: number of action classes (default 3)
        hidden_dim:  LSTM hidden size per direction (default 128)

    Returns:
        SportActionLSTM model instance
    """
    return SportActionLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                           num_layers=2, num_classes=num_classes)
