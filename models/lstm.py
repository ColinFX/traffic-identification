import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO URGENT input_size and out_features passed by parameters
        self.lstm = nn.LSTM(input_size=54, hidden_size=256, num_layers=5, batch_first=True)
        self.fc1 = nn.Linear(in_features=256, out_features=9)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(5, data_batch.shape[0], 256).to(device=data_batch.device)
        c0 = torch.zeros(5, data_batch.shape[0], 256).to(device=data_batch.device)
        data_batch, _ = self.lstm(data_batch, (h0, c0))
        data_batch = self.fc1(data_batch[:, -1, :])
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