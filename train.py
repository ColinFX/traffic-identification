import argparse
import logging
import os
from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataloader import SrsRANLteDataLoaders
from evaluate import evaluate
from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier
from models.transformer import TransformerEncoderClassifier, TransformerLSTMClassifier
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/srsRAN/srsenb0219")
parser.add_argument("--experiment_dir", default="experiments/trial-59/78")  # hyper-parameter json file
parser.add_argument("--restore_file", default=None)
# pretrained model weight checkpoint for second stage of two-stage training
parser.add_argument("--zero_shot_keyword", default="78")  # scenario as val and test set for zero-shot


def train_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor],
        data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor]],
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
            train_batch = train_batch.cuda(device=torch.device(params.cuda_index))
            true_labels_batch = true_labels_batch.cuda(device=torch.device(params.cuda_index))
        predicted_proba_batch: torch.Tensor = model(train_batch)
        train_loss = loss_fn(predicted_proba_batch, true_labels_batch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # evaluate summaries once in a while within one epoch
        if step % params.save_summary_steps == 0:
            predicted_proba_batch = predicted_proba_batch.detach().cpu().numpy()
            true_labels_batch = true_labels_batch.detach().cpu().numpy()
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
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor]],
        params: utils.HyperParams,
        experiment_dir: str
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
        * experiment_dir: (str) directory containing config, checkpoints and log
    Returns:
        * metrics: (dict) metric_name -> metric_value on the val set of the best epoch
    """
    # reload weights from checkpoint is available
    if args.restore_file and os.path.isfile(os.path.join(args.experiment_dir, args.restore_file + ".pth.tar")):
        restore_path = os.path.join(args.experiment_dir, args.restore_file + ".pth.tar")
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
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)
        val_acc = val_metrics["accuracy"]
        is_best = (val_acc >= best_val_acc)

        # save weights checkpoint
        utils.save_checkpoint(
            state={"epoch": epoch + 1, "state_dict": model.state_dict(), "optim_dict": optimizer.state_dict()},
            is_best=is_best,
            checkpoint_dir=experiment_dir
        )

        # overwrite the best metrics evaluation result if the models is the best by far
        if is_best:
            best_metrics = val_metrics
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            best_json_path = os.path.join(experiment_dir, "metrics_val_best_weights.json")
            utils.save_metrics(val_metrics, best_json_path)

        # overwrite last metrics evaluation result
        last_json_path = os.path.join(experiment_dir, "metric_val_last_weights.json")
        utils.save_metrics(val_metrics, last_json_path)

    return best_metrics


if __name__ == "__main__":
    """Train the models on the train and validation set"""
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
    utils.set_logger(os.path.join(args.experiment_dir, "train.log"))
    logging.info("Loading the dataset...")

    # load data
    all_paths = utils.listdir_with_suffix(args.data_dir, ".npz")
    train_npz_paths = [path for path in all_paths if "_10" in path and args.zero_shot_keyword not in path]
    val_test_npz_paths = [path for path in all_paths if "_10" in path and args.zero_shot_keyword in path]
    dataloaders = SrsRANLteDataLoaders(
        params=params,
        split_percentages=[0, 0.75, 0.25],
        read_train_npz_paths=train_npz_paths,
        read_val_test_npz_paths=val_test_npz_paths,
        label_mapping=utils.srsRANLte_label_mapping,
        # save_npz_path=os.path.join(args.experiment_dir, "train_save.npz")
    )
    dataloaders.save_label_encoder(os.path.join(args.experiment_dir, "label_encoder.pkl"))

    train_dataloader = dataloaders.train
    val_dataloader = dataloaders.val
    params.train_size = len(train_dataloader.dataset)
    params.val_size = len(val_dataloader.dataset)

    # prepare model
    classifier = TransformerEncoderClassifier(
        raw_embedding_len=59,
        sequence_length=10,
        num_classes=6,
        target_embedding_len=params.transformer_target_embedding_len,
        transformer_num_head=params.transformer_num_head,
        transformer_dimension_feedforward=params.transformer_dimension_feedforward,
        transformer_dropout=params.transformer_dropout,
        transformer_activation=params.transformer_activation,
        transformer_num_layers=params.transformer_num_layers,
        lstm_hidden_size=params.lstm_hidden_size,
        lstm_num_layers=params.lstm_num_layers,
        lstm_dropout=params.lstm_dropout,
        lstm_bidirectional=params.lstm_bidirectional,
        upstream_model=params.upstream_model,
        downstream_model=params.downstream_model
    )

    # Stage 2 for two-stage training:
    # pretrained_transformer_encoder = TransformerEncoderClassifier(
    #     raw_embedding_len=59,
    #     sequence_length=10,
    #     num_classes=6,
    #     target_embedding_len=params.transformer_target_embedding_len,
    #     transformer_num_head=params.transformer_num_head,
    #     transformer_dimension_feedforward=params.transformer_dimension_feedforward,
    #     transformer_dropout=params.transformer_dropout,
    #     transformer_activation=params.transformer_activation,
    #     transformer_num_layers=params.transformer_num_layers,
    #     lstm_hidden_size=params.lstm_hidden_size,
    #     lstm_num_layers=params.lstm_num_layers,
    #     lstm_dropout=params.lstm_dropout,
    #     lstm_bidirectional=params.lstm_bidirectional,
    #     upstream_model=params.upstream_model,
    #     downstream_model=params.downstream_model
    # )
    # utils.load_checkpoint(
    #     os.path.join(args.restore_file_path),
    #     pretrained_transformer_encoder
    # )
    # lstm_classifier = LSTMClassifier(
    #     embedding_len=params.transformer_target_embedding_len,
    #     num_classes=6,
    #     num_layers=params.lstm_num_layers,
    #     dropout=params.lstm_dropout,
    #     bidirectional=params.lstm_bidirectional
    # )
    # classifier = TransformerLSTMClassifier(
    #     pretrained_transformer_encoder=pretrained_transformer_encoder,
    #     lstm_classifier=lstm_classifier
    # )

    # train
    if params.cuda_index > -1:
        classifier.cuda(device=torch.device(params.cuda_index))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    train(
        model=classifier,
        optimizer=optimizer,
        loss_fn=utils.loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metrics=utils.metrics,
        params=params,
        experiment_dir=args.experiment_dir
    )
