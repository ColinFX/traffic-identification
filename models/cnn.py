import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.mp1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.mp2 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.fc1 = nn.Linear(in_features=32*4*17, out_features=2)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            * data_batch: (torch.FloatTensor) contains a batch of images, shape: batch_size * 30 * 80
        Returns:
            * out: (torch.FloatTensor) predicted log probability distribution of each image, shape: batch_size * 2
        """
        # input: batch_size * 30 * 80
        data_batch = data_batch.view((data_batch.shape[0], 1, 30, 80))  # batch_size * 1 * 30 * 80
        data_batch = self.conv1(data_batch)  # batch_size * 16 * 26 * 76
        data_batch = F.relu(data_batch)
        data_batch = self.mp1(data_batch)  # batch_size * 16 * 13 * 38
        data_batch = self.conv2(data_batch)  # batch_size * 32 * 9 * 34
        data_batch = F.relu(data_batch)
        data_batch = self.mp2(data_batch)  # batch_size * 32 * 4 * 17
        data_batch = data_batch.view(-1, 32*4*17)  # batch_size * 32*4*17
        data_batch = self.fc1(data_batch)  # batch_size * 2
        data_batch = F.softmax(data_batch, dim=1)
        return data_batch  # output: batch_size * 2

    def reset_weights(self):
        """Reset all weights before next fold to avoid weight leakage"""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


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