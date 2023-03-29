import argparse
import logging
import os
from typing import Callable, Iterator

import numpy as np
import torch
from tqdm import trange

import models
import utils
from models.dataloader import TCDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--experiment_dir", default="experiments/base")  # hyper-parameter json file
parser.add_argument("--restore_file", default="best")  # "best" or "last", models weights checkpoint


def evaluate(
        model: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor],
        data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
        num_steps: int
) -> dict[str, float]:
    """
    Evaluate the models on `num_steps` batches/iterations of size `params.batch_size` as one epoch.
    Args:
        * model: (nn.Module) the neural network
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * data_iterator: (Generator) -> train_batch, labels_batch
        * metrics: (dict) metric_name -> (function (Callable) predicted_proba_batch, true_labels_batch -> metric_value)
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
            eval_batch.cuda(device=torch.device(params.cuda_index))
        predicted_proba_batch = model(eval_batch)
        eval_loss = loss_fn(predicted_proba_batch, true_labels_batch)

        # evaluate all metrics on every batch
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
    json_path = os.path.join(args.model_dir, "params.json")
    if not os.path.isfile(json_path):
        raise("Failed to load hyperparameters: no json file found at {}.".format(json_path))
    params = utils.HyperParams(json_path)

    # bypass cuda index hyperparameter if specified cuda device is not available
    if params.cuda_index >= torch.cuda.device_count():
        params.cuda_index = -1

    # set random seed for reproducibility
    torch.manual_seed(42)
    if params.cuda_index > -1:
        torch.cuda.manual_seed(42)

    # set logger
    utils.set_logger(os.path.join(args.model_dir, "test.log"))
    logging.info("Loading the dataset...")

    # load data
    test_data_loader = TCDataLoader("test", args.data_dir, params)
    params.test_size = len(test_data_loader.dataset)
    num_steps = (params.test_size+1) // params.batch_size

    # evaluate pipeline
    cnn = models.CNN.CNN()
    if params.cuda_index:
        cnn.cuda(device=torch.device(params.cuda_index))
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file+".pth.tar"), cnn)
    test_metrics = evaluate(
        model=cnn,
        loss_fn=models.CNN.loss_fn,
        data_iterator=iter(test_data_loader),
        metrics=models.CNN.metrics,
        num_steps=num_steps
    )

    # save metrics evaluation result on the restore_file
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_metrics_to_json(test_metrics, save_path)
