import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


class HyperParams:
    """
    Examples of usage:
        * create params instance by `params = HyperParams("./params.json")`
        * change param value by `params.learning_rate = 0.5`
        * show param value by `print(params.learning_rate)`
        * save params instance by `params.save("./params.json")`
    """
    batch_size: int
    cuda_index: int
    expansion_factor: int
    learning_rate: float
    n_heads: int
    num_epochs: int
    num_layers: int
    pca_n_components: int
    random_seed: int
    re_preprocess: bool
    save_summary_steps: int
    split_test_percentage: float
    split_val_percentage: float
    tb_len_threshold: int
    window_size: int
    train_size: int
    val_size: int
    test_size: int

    def __init__(self, json_path: str):
        with open(json_path, 'r') as file:
            params = json.load(file)
            self.__dict__.update(params)

    def save(self, json_path: str):
        with open(json_path, 'w') as file:
            json.dump(self.__dict__, file, indent=4)

    def update(self, json_path: str):
        with open(json_path, 'r') as file:
            params = json.load(file)
            self.__dict__.update(params)

    @property
    def dict(self):
        """dict-like access to param by `params.dict["learning_rate"]`"""
        return self.__dict__


class RunningAverage:
    """
    Examples of usage:
        * create running_avg instance by `running_avg = RunningAverage()`
        * add new item by `running_avg.update(2.0)`
        * get current average by `running_avg()`
    """

    def __init__(self):
        self.steps: int = 0
        self.total: float = 0

    def update(self, val: float):
        self.steps += 1
        self.total += val

    def __call__(self) -> float:
        return self.total / float(self.steps)


def set_logger(log_path: str):
    """
    Examples of usage:
        * initialize logger by `set_logger("./models/train.log")`
        * and then write log by `logging.info("Start training...")`
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # logging to the file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_metrics(dictionary: dict[str, float], json_path: str):
    """Save dictionary containing metrics results to json file"""
    with open(json_path, 'w') as file:
        # convert the values to float for json since it doesn't accept np.array, np.float, nor torch.FloatTensor
        dictionary = {key: float(value) for key, value in dictionary.items()}
        json.dump(dictionary, file, indent=4)


def save_checkpoint(state: dict[str, float or dict], is_best: bool, checkpoint_dir: str):
    """
    Save checkpoint to designated directory, creating directory if not exist.
    Args:
        * state: (dict) containing "models" key and maybe "epoch" and "optimizer"
          is a python dictionary object that maps each layer to its parameter tensor
        * is_best: (bool) whether the model is the best till that moment
        * checkpoint_dir: (str) folder to save weights
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.save(state, os.path.join(checkpoint_dir, "last.pth.tar"))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pth.tar"))


def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None
) -> dict[str, float or dict]:
    """
    Args:
        * checkpoint_path: (str) path of the checkpoint
        * model: (nn.Module) models that weights will be loaded to
        * optimizer: (torch.optim.Optimizer) optional - optimizer that weights will be loaded to
    """
    if not os.path.exists(checkpoint_path):
        raise ("Failed to load checkpoint: file does not exist {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint


def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
    """
    Args:
        * outputs: (torch.FloatTensor) output of the model, shape: batch_size * 2
        * labels: (torch.Tensor) ground truth label of the image, shape: batch_size with each element a value in [0, 1]
    Returns:
        * loss: (torch.FloatTensor) cross entropy loss for all images in the batch
    """
    loss = torch.nn.CrossEntropyLoss()
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


def rough_eval(string: str) -> float:
    try:
        return eval(string)
    except (NameError, TypeError, SyntaxError) as _:
        return -1


srsRANLte_channels: List[str] = ["PUSCH", "PDSCH", "PUCCH", "PDCCH", "PHICH"]


amariNSA_channels: Dict[str, List[str]] = {
    "03": ["PUSCH", "PDSCH", "PUCCH", "PDCCH", "SRS", "PHICH"],
    "04": ["PUSCH", "PDSCH", "PUCCH", "PDCCH"]
}
