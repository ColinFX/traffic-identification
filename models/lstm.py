import numpy as np
import torch
import torch.nn as nn

import utils


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            embedding_length: int,
            num_classes: int,
            hidden_size: int = 256,
            num_layers: int = 3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_features = num_classes
        self.lstm = nn.LSTM(
            input_size=embedding_length,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.out_features)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        # batch_size * sequence_length * embedding_length
        data_batch = data_batch.transpose(0, 1)  # sequence_length * batch_size * embedding_length
        data_batch, _ = self.lstm(data_batch)  # sequence_length * batch_size * hidden_size
        data_batch = self.fc1(data_batch[-1, :, :])  # batch_size * num_classes
        return data_batch

    def reset_weights(self):
        """Reset all weights before next fold to avoid weight leakage"""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


# TODO: move this to utils
def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
    """
    Args:
        * outputs: (torch.FloatTensor) output of the model, shape: batch_size * 2
        * labels: (torch.Tensor) ground truth label of the image, shape: batch_size with each element a value in [0, 1]
    Returns:
        * loss: (torch.FloatTensor) cross entropy loss for all images in the batch
    """
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)


def accuracy(outputs: np.ndarray[np.float32], labels: np.ndarray[np.int64]) -> np.float64:
    """
    Args:
        * outputs: (np.ndarray) outpout of the model, shape: batch_size * 2
        * labels: (np.ndarray) ground truth label of the image, shape: batch_size with each element a value in [0, 1]
    Returns:
        * accuracy: (float) in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


metrics = {"accuracy": accuracy}
