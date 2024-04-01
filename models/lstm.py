import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            embedding_len: int,
            num_classes: int,
            hidden_size: int = 256,
            num_layers: int = 5,
            dropout: float = 0.0,
            bidirectional: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=embedding_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        if bidirectional:
            self.fc1 = nn.Linear(in_features=2*hidden_size, out_features=num_classes)
        else:
            self.fc1 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        # data_batch: batch_size * sequence_length * embedding_length
        data_batch, _ = self.lstm(data_batch)
        # data_batch: batch_size * sequence_length * embedding_length
        data_batch = self.fc1(data_batch[:, -1, :])
        # data_batch: batch_size * num_classes
        return data_batch

    def reset_weights(self):
        """Reset all weights before next fold to avoid weight leakage"""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
