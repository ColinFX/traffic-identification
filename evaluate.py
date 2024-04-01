import argparse
import datetime
import logging
import os
import pickle
from typing import Callable, Iterator

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from dataloader import AmariNSADataLoaders, SrsRANLteDataLoaders
from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier
from models.transformer import TransformerEncoderClassifier
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/srsRAN/srsenb0219")
parser.add_argument("--experiment_dir", default="experiments/trial-51")  # hyper-parameter json file
parser.add_argument("--restore_file", default="best")  # "best" or "last", models weights checkpoint


def evaluate(
        model: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor],
        data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor]],
        params: utils.HyperParams,
        num_steps: int
) -> dict[str, float]:
    """
    Evaluate the models on `num_steps` batches/iterations of size `params.batch_size` as one epoch.
    Args:
        * model: (nn.Module) the neural network
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * data_iterator: (Generator) -> train_batch, labels_batch
        * metrics: (dict) metric_name -> (function (Callable) predicted_proba_batch, true_labels_batch -> metric_value)
        * params: (utils.Params) hyperparameters
        * num_steps: (int) number of batches to train for each epoch
    Returns:
        * metric_results: (dict) metric_name -> metric_value, metrics are provided metrics and loss
    """
    model.eval()  # set models to evaluation mode
    summary: list[dict[str, float]] = []  # summary of metrics for the epoch

    t = trange(num_steps)
    for _ in t:
        # core pipeline
        eval_batch, true_labels_batch = next(data_iterator)
        if params.cuda_index > -1:
            eval_batch = eval_batch.cuda(device=torch.device(params.cuda_index))
        predicted_proba_batch = model(eval_batch)
        predicted_proba_batch = predicted_proba_batch.detach().cpu()
        eval_loss = loss_fn(predicted_proba_batch, true_labels_batch)

        # evaluate all metrics on every batch
        predicted_proba_batch = predicted_proba_batch.numpy()
        true_labels_batch = true_labels_batch.detach().cpu().numpy()
        batch_summary = {metric: metrics[metric](predicted_proba_batch, true_labels_batch) for metric in metrics}
        batch_summary["eval_loss"] = eval_loss.item()
        summary.append(batch_summary)

    metrics_mean = {metric: np.mean([batch[metric] for batch in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(key, value) for key, value in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean


if __name__ == "__main__":
    """Evaluate the model on the test set"""
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, "params.json")
    if not os.path.isfile(json_path):
        raise ("Failed to load hyperparameters: no json file found at {}.".format(json_path))
    params = utils.HyperParams(json_path)

    # bypass cuda index hyperparameter if specified cuda device is not available
    if params.cuda_index >= torch.cuda.device_count():
        params.cuda_index = -1

    # set random seed for reproducibility
    torch.manual_seed(params.random_seed)
    if params.cuda_index > -1:
        torch.cuda.manual_seed(params.random_seed)

    # set logger
    utils.set_logger(os.path.join(args.experiment_dir, "test.log"))
    logging.info("Loading the dataset...")

    # load data
    # Option 1: load data from pre-generated npz file
    # npz_dict = np.load(os.path.join(args.experiment_dir, "train_save.npz"))
    # test_dataset = TensorDataset(
    #     torch.Tensor(npz_dict["X_test"]),
    #     torch.Tensor(npz_dict["y_test"]).type(torch.long)
    # )
    # test_dataloader = DataLoader(test_dataset, params.batch_size, shuffle=False)

    # Option 2: instantiate new SrsRANLteData
    label_mapping = {}
    for gain in [66, 69, 72, 75, 78, 81, 84]:
        for app in ["bililive", "bilivideo", "netdisk", "tmeetingaudio", "tmeetingvideo", "wget"]:
            label_mapping[app + str(gain)] = app
            label_mapping[app + str(gain) + "_10"] = app
    all_paths = utils.listdir_with_suffix(args.data_dir, ".npz")
    val_test_npz_paths = []
    for path in all_paths:
        if "_10" in path and "81" in path:
            val_test_npz_paths.append(path)
    with open(os.path.join(args.experiment_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder: LabelEncoder = pickle.load(f)
    dataloaders = SrsRANLteDataLoaders(
        params=params,
        split_percentages=[0, 0, 1],
        read_val_test_npz_paths=val_test_npz_paths,
        label_mapping=label_mapping,
        label_encoder=label_encoder,
        save_npz_path=os.path.join(args.experiment_dir, "test_save.npz")
    )
    test_dataloader = dataloaders.test

    # evaluate pipeline
    # classifier = LSTMClassifier(
    #     embedding_len=59,
    #     num_classes=6
    # )
    classifier = TransformerEncoderClassifier(
        raw_embedding_len=59,
        sequence_length=10,
        num_classes=6,
        upstream_model="linear",
        downstream_model="linear"
    )

    if params.cuda_index > -1:
        classifier.cuda(device=torch.device(params.cuda_index))
    utils.load_checkpoint(os.path.join(args.experiment_dir, args.restore_file + ".pth.tar"), classifier)
    test_metrics = evaluate(
        model=classifier,
        loss_fn=utils.loss_fn,
        data_iterator=iter(test_dataloader),
        metrics=utils.metrics,
        params=params,
        num_steps=len(test_dataloader)
    )

    # save metrics evaluation result on the restore_file
    save_path = os.path.join(args.experiment_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_metrics(test_metrics, save_path)
