"""
Phase 2 — E2 GRU Model
========================
Locked architecture (sequence_specification.md v1.1):
  Input  : [batch, 4, 406]  — 4 history steps, 406 features
  GRU    : input_size=406, hidden_size=64, num_layers=1, dropout=0.2
  Output : Linear(64 → 1) returning LOGITS

IMPORTANT — logits vs probabilities:
  During training  : model returns raw logits → consumed by BCEWithLogitsLoss
  During inference : sigmoid(logits) → fraud probability in [0, 1]
  Sigmoid is NOT applied inside the model to preserve numerical stability
  with BCEWithLogitsLoss (which applies log-sigmoid internally).

ISOLATION: no imports from src/ (E1). All code is self-contained in phase2_gru/.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FraudGRU(nn.Module):
    """
    Single-layer GRU for transaction sequence fraud detection.

    Architecture (locked):
        GRU(input_size=406, hidden_size=64, num_layers=1,
            batch_first=True, dropout=0.0)
        → take final hidden state h_T
        → Dropout(0.2)
        → Linear(64, 1)
        → logits (NOT probabilities)

    Note on dropout:
        PyTorch GRU applies dropout between layers (i.e., on inputs to
        layers 2..num_layers). With num_layers=1 there is no inter-layer
        dropout available inside the GRU cell itself. The locked 0.2 dropout
        is therefore applied as a standalone Dropout layer after the final
        hidden state, before the linear classifier. This is equivalent to the
        locked specification and is standard practice for single-layer GRUs.

    Args:
        input_size  : feature dimension per time step (default 406)
        hidden_size : GRU hidden state dimension (default 64)
        num_layers  : number of stacked GRU layers (default 1)
        dropout     : dropout rate applied after final hidden state (default 0.2)
    """

    def __init__(
        self,
        input_size: int = 406,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout

        # GRU layer
        # batch_first=True  → input shape [batch, seq_len, features]
        # dropout inside GRU is only meaningful for num_layers > 1
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,          # inter-layer; irrelevant for num_layers=1
        )

        # Post-GRU dropout (applies locked 0.2 rate to final hidden state)
        self.dropout = nn.Dropout(p=dropout)

        # Output classifier — returns LOGITS (no sigmoid)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : FloatTensor of shape [batch_size, seq_len, input_size]
                Expected: [batch, 4, 406]

        Returns:
            logits : FloatTensor of shape [batch_size, 1]
                     Raw logits (NOT sigmoid-transformed).
                     Use sigmoid(logits) to get fraud probabilities.
        """
        # GRU forward
        # output : [batch, seq_len, hidden_size]
        # h_n    : [num_layers, batch, hidden_size]
        output, h_n = self.gru(x)

        # Take the final hidden state of the last layer
        # h_n[-1] : [batch, hidden_size]
        final_hidden = h_n[-1]

        # Apply dropout
        final_hidden = self.dropout(final_hidden)

        # Linear projection → logits [batch, 1]
        logits = self.classifier(final_hidden)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference: returns fraud probabilities in [0, 1].

        Args:
            x : FloatTensor [batch, 4, 406]

        Returns:
            proba : FloatTensor [batch, 1] in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self) -> dict[str, int]:
        """Return per-layer and total trainable parameter counts."""
        counts = {}
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                n = param.numel()
                counts[name] = n
                total += n
        counts["_total"] = total
        return counts

    def __repr__(self) -> str:
        return (
            f"FraudGRU("
            f"input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout_p})"
        )
