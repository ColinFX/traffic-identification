import argparse
import logging
import os
from typing import Callable, Iterator

import numpy as np
import torch
from tqdm import trange

import models
import utils
from evaluate import evaluate
from models.dataloader import TCDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--experiment_dir", default="experiments/base")  # hyper-parameter json file
parser.add_argument("--restore_file", default=None)  # "best" or "last", models weights checkpoint


def train_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor],
        data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
        params: utils.HyperParams,
        num_steps: int
):
    """
    Train the models on `num_steps` batches/iterations of size `params.batch_size` as one epoch.
    Args:
        * model: (nn.Module) the neural network
        * optimizer: (torch.optim.Optimizer) the optimizer for parameters in the models
        * loss_fn: (Callable) predicted_proba_batch, true_labels_batch -> train_loss
        * data_iterator: (Generator) -> train_batch, true_labels_batch
        * metrics: (dict) metric_name -> (function (Callable) predicted_proba_batch, true_labels_batch -> metric_value)
        * params: (utils.HyperParams) hyperparameters
        * num_steps: (int) number of batches to train for each epoch
    """
    model.train()
    summary: list[dict[str, float]] = []
    loss_avg = utils.RunningAverage()

    t = trange(num_steps)
    for step in t:
        # core pipeline
        train_batch, true_labels_batch = next(data_iterator)
        if params.cuda_index > -1:
            train_batch.cuda(device=torch.device(params.cuda_index))
            true_labels_batch.cuda(device=torch.device(params.cuda_index))
        predicted_proba_batch = model(train_batch)
        train_loss = loss_fn(predicted_proba_batch, true_labels_batch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # evaluate summaries once in a while within one epoch
        if step % params.save_summary_steps == 0:
            batch_summary = {metric: metrics[metric](predicted_proba_batch, true_labels_batch) for metric in metrics}
            batch_summary["train_loss"] = train_loss.item()
            summary.append(batch_summary)

        loss_avg.update(train_loss.item())
        t.set_postfix(train_loss="{:05.3f}".format(loss_avg()))

    metrics_mean = {metric: np.mean([batch[metric] for batch in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(key, value) for key, value in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor],
        train_dataloader: TCDataLoader,
        val_dataloader: TCDataLoader,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
        params: utils.HyperParams,
        model_dir: str
) -> dict[str, float]:
    """
    Train and evaluate the models on `params.num_epochs` epochs, save checkpoints and metrics.
    Args:
        * model: (nn.Module) the neural network
        * optimizer: (torch.optim.Optimizer) the optimizer for parameters in the models
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * train_dataloader: (TCDataLoader) for training set
        * val_dataloader: (TCDataLoader) for validation set
        * metrics: (dict) metric_name -> (function (Callable) output_batch, labels_batch -> metric_value)
        * params: (utils.Params) hyperparameters
        * model_dir: (str) directory containing config, checkpoints and log
    Returns:
        * metrics: (dict) metric_name -> metric_value on the val set of the best epoch
    """
    # reload weights from checkpoint is available
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + ".pth.tar")
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    best_metrics: dict[str, float] = {}

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # train
        num_steps = int((params.train_size + 1) // params.batch_size)
        train_data_iterator = iter(train_dataloader)
        train_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            data_iterator=train_data_iterator,
            metrics=metrics,
            params=params,
            num_steps=num_steps
        )

        # evaluate
        num_steps = int((params.val_size + 1) // params.batch_size)
        val_data_iterator = iter(val_dataloader)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, num_steps)
        val_acc = val_metrics["accuracy"]
        is_best = (val_acc >= best_val_acc)

        # save weights checkpoint
        utils.save_checkpoint(
            state={"epoch": epoch + 1, "state_dict": model.state_dict(), "optim_dict": optimizer.state_dict()},
            is_best=is_best,
            checkpoint_dir=model_dir
        )

        # overwrite the best metrics evaluation result if the models is the best by far
        if is_best:
            best_metrics = val_metrics
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_metrics_to_json(val_metrics, best_json_path)

        # overwrite last metrics evaluation result
        last_json_path = os.path.join(model_dir, "metric_val_last_weights.json")
        utils.save_metrics_to_json(val_metrics, last_json_path)

    return best_metrics


if __name__ == "__main__":
    """Train the models on the train and validation set"""
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
    utils.set_logger(os.path.join(args.model_dir, "train.log"))
    logging.info("Loading the dataset...")

    # load data
    train_dataloader = TCDataLoader("train", args.data_dir, params)
    val_dataloader = TCDataLoader("val", args.data_dir, params)
    params.train_size = len(train_dataloader.dataset)
    params.val_size = len(val_dataloader.dataset)

    # train pipeline
    cnn = models.CNN.CNN()
    if params.cuda_index:
        cnn.cuda(device=torch.device(params.cuda_index))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=params.learning_rate)
    train(
        model=cnn,
        optimizer=optimizer,
        loss_fn=models.CNN.loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metrics=models.CNN.metrics,
        params=params,
        model_dir=args.model_dir
    )
