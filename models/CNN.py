from typing import Callable

import numpy as np
import sklearn
import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(in_features=7*7*32, out_features=10)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        # batch_size * 1 * 28 * 28
        data_batch = self.conv1(data_batch)  # batch_size * 16 * 28 * 28
        data_batch = torch.nn.functional.relu(data_batch)
        data_batch = self.mp1(data_batch)  # batch_size * 16 * 14 * 14
        data_batch = self.conv2(data_batch)  # batch_size * 32 * 14 * 14
        data_batch = torch.nn.functional.relu(data_batch)
        data_batch = self.mp2(data_batch)  # batch_size * 16 * 7 * 7

        # flatten
        data_batch = data_batch.view(-1, 7 * 7 * 32)  # batch_size * (7*7*16)
        data_batch = self.fc1(data_batch)  # batch_size * 10
        return data_batch

    def reset_weights(self):
        """Reset all weights before next fold to avoid weight leakage"""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def loss_fn(predicted_proba: torch.Tensor, true_labels: torch.Tensor) -> torch.FloatTensor:
    """Keep at the computing device and do not detach"""
    loss = torch.nn.CrossEntropyLoss()
    return loss(predicted_proba, true_labels)


def accuracy(predicted_proba: torch.Tensor, true_labels: torch.Tensor) -> float:
    predicted_proba = predicted_proba.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()
    predict = np.argmax(predicted_proba, axis=1)
    return sklearn.metrics.accuracy_score(true_labels, predict)


metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {"accuracy": accuracy}
