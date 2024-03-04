import json
import logging
import os
from typing import Dict, List
import warnings

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

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


class AdvancedMinMaxScaler:
    """
    Modified version of vanilla sklearn.preprocessing.MinMaxScaler allowing `fit` multiple times on different parts of
    the whole dataset. Instance arguments and all other methods are kept the same usage.
    """
    def __init__(self, **kwargs):
        self.data_max: np.ndarray = np.empty(0)
        self.data_min: np.ndarray = np.empty(0)
        self.ever_fit = False
        self.kwargs = kwargs

    def _form_fake_X(self) -> np.ndarray:
        return np.stack((self.data_max, self.data_min))

    def fit(self, X):
        new_scaler = MinMaxScaler(**self.kwargs)
        new_scaler.fit(X)
        if not self.ever_fit:
            self.data_max = new_scaler.data_max_
            self.data_min = new_scaler.data_min_
            self.ever_fit = True
        else:
            if len(self.data_max) != new_scaler.n_features_in_:
                raise ValueError(
                    "X has {} features, but MinMaxScalerAdvanced is expecting {} features as input.".format(
                        new_scaler.n_features_in_,
                        len(self.data_max)
                    )
                )
            self.data_max = np.maximum(self.data_max, new_scaler.data_max_)
            self.data_min = np.minimum(self.data_min, new_scaler.data_min_)

    def transform(self, X) -> np.ndarray:
        if not self.ever_fit:
            raise NotFittedError(
                "This MinMaxScalerAdvanced instance is not fit yet. "
                "Call 'fit' at least once before using this estimator."
            )
        scaler = MinMaxScaler(**self.kwargs)
        scaler.fit(self._form_fake_X())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # fit on fake_X without feature names
            transformed_X = scaler.transform(X)
        return transformed_X

    def fit_transform(self, X) -> np.ndarray:
        scaler = MinMaxScaler(**self.kwargs)
        return scaler.fit_transform(X)


class AdvancedOneHotEncoder:
    """
    Modified version of vanilla sklearn.preprocessing.OneHotEncoder allowing `fit` multiple times on different parts of
    the whole dataset. Instance arguments and all other methods are kept the same usage.
    """

    def __init__(self, **kwargs):
        self.categories: List[np.ndarray] = []
        self.ever_fit = False
        self.kwargs = kwargs

    def _form_fake_X(self):
        fake_X = []
        max_feature_categories_num = np.max([len(feature_categories) for feature_categories in self.categories])
        for category_idx in range(max_feature_categories_num):
            fake_X.append(
                [
                    self.categories[feature_idx][min(category_idx, len(self.categories[feature_idx])-1)]
                    for feature_idx in range(len(self.categories))
                ]
            )
        return fake_X

    def fit(self, X):
        new_encoder = OneHotEncoder(**self.kwargs)
        new_encoder.fit(X)
        if not self.ever_fit:
            self.categories = new_encoder.categories_
            self.ever_fit = True
        else:
            if len(self.categories) != new_encoder.n_features_in_:
                raise ValueError(
                    "X has {} features, but OneHotEncoderAdvanced is expecting {} features as input.".format(
                        new_encoder.n_features_in_,
                        len(self.categories)
                    )
                )
            for feature_idx in range(len(self.categories)):
                for new_category in new_encoder.categories_[feature_idx]:
                    if new_category not in self.categories[feature_idx]:
                        self.categories[feature_idx] = np.append(self.categories[feature_idx], new_category)

    def transform(self, X):
        if not self.ever_fit:
            raise NotFittedError(
                "This OneHotEncoderAdvanced instance is not fit yet. "
                "Call 'fit' at least once before using this estimator."
            )
        encoder = OneHotEncoder(**self.kwargs)
        encoder.fit(self._form_fake_X())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # fit on fake_X without feature names
            encoded_X = encoder.transform(X)
        return encoded_X

    def fit_transform(self, X):
        encoder = OneHotEncoder(**self.kwargs)
        return encoder.fit_transform(X)

    def get_features_num(self):
        if self.ever_fit:
            encoder = OneHotEncoder(**self.kwargs)
            encoder.fit(self._form_fake_X())
            return len(encoder.get_feature_names_out())
        else:
            return 0


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


def listdir_with_suffix(parent_dir: str, suffix: str):
    legitimate_paths: List[str] = []
    for file_name in os.listdir(parent_dir):
        if os.path.splitext(file_name)[1] == suffix:
            legitimate_paths.append(os.path.join(parent_dir, file_name))
    return legitimate_paths


srsRANLte_channels: List[str] = ["PUSCH", "PDSCH", "PUCCH", "PDCCH", "PHICH"]


amariNSA_channels: Dict[str, List[str]] = {
    "03": ["PUSCH", "PDSCH", "PUCCH", "PDCCH", "SRS", "PHICH"],
    "04": ["PUSCH", "PDSCH", "PUCCH", "PDCCH"]
}

