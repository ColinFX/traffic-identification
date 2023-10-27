import argparse
import datetime
import logging
import os
import time
from typing import Dict, List, Tuple

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

import utils
from dataloader import AmariNSADataset, SrsRANLteDataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data/srsRAN/srsenb1009")
parser.add_argument("--experiment_dir", default="../experiments/base")  # hyper-parameter json file


def get_data(params: utils.HyperParams) -> Tuple[np.ndarray, np.ndarray]:
    """Read dataset from npz file or preprocess from raw log file"""
    if params.re_preprocess:
        dataset = AmariNSADataset(
            params=params,
            feature_path=os.path.join(args.experiment_dir, "features.json"),
            read_log_paths=[os.path.join(args.data_dir, file) for file in ["gnb0.log"]],
            timetables=[[
                ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
                ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
            ]],
            save_path=os.path.join(args.data_dir, "dataset_Xy.npz")
        )
    else:
        dataset = SrsRANLteDataset(
            params=params,
            read_npz_path=os.path.join(args.data_dir, "dataset_Xy.npz")
        )
        # dataset = AmariNSADataset(
        #     params=params,
        #     feature_path=os.path.join(args.experiment_dir, "features.json"),
        #     read_npz_path=os.path.join(args.data_dir, "dataset_Xy.npz")
        # )
    dataset.X = np.reshape(dataset.X, (dataset.X.shape[0], -1))
    return dataset.X, dataset.y


def model_selection(X: np.ndarray, y: np.ndarray, params: utils.HyperParams) -> Dict[str, any]:
    """Fit and test multiple machine learning models without hyperparameter tuning for rough model selection"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.split_test_percentage,
        random_state=params.random_seed
    )
    models = {
        "sgd": SGDClassifier(),
        # "svc": SVC(),
        "rf": RandomForestClassifier(),
        # "mlp": MLPClassifier(),
        "tree": ExtraTreeClassifier(),
        "xgb": XGBClassifier(),
        "cat": CatBoostClassifier(allow_writing_files=False, verbose=False),
        "lgb": lgb.LGBMClassifier()
    }
    for model_name in models.keys():
        model = models[model_name]
        train_start = time.time()
        model.fit(X_train, y_train)
        logging.info("Training of model {} over {} records completed in {} seconds".format(
            model_name, y_train.shape[0], time.time() - train_start
        ))
        test_start = time.time()
        y_test_pred = model.predict(X_test)
        logging.info("Testing of model {} over {} records completed in {} seconds".format(
            model_name, y_test.shape[0], time.time() - test_start
        ))
        logging.info(">> {:<3} {:.4f} {:.4f}".format(
            model_name, accuracy_score(y_test, y_test_pred), f1_score(y_test, y_test_pred, average="macro")
        ))
        # logging.info(">> {:<3} {:.4f} {:.4f}".format(
        #     model_name, accuracy_score(y_test, y_test_pred), log_loss(y_test, y_test_pred)
        # ))
        logging.info(confusion_matrix(y_test, y_test_pred))
    return models


def _objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray):
    """Inner objective function for optuna hyperparameter tuning"""
    param_grid = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 5000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 500, 1000, 3000]),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.random_seed)
    cv_scores = np.empty(5)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx, :], X[val_idx, :]
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=(
                [optuna.integration.LightGBMPruningCallback(trial, "binary_logloss"), lgb.log_evaluation(0)]
                if idx == 0
                else [lgb.log_evaluation(0)]
            ),
        )
        y_eval_pred_proba = model.predict_proba(X_val)
        cv_scores[idx] = log_loss(y_val, y_eval_pred_proba)
    return np.mean(cv_scores)


def lgb_tuning(X: np.ndarray, y: np.ndarray, params: utils.HyperParams) -> lgb.LGBMClassifier:
    """Hyperparameter tuning of LGBMClassifier model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.split_test_percentage,
        random_state=params.random_seed
    )
    study = optuna.create_study(direction="minimize", study_name="LGBMClassifier")
    study.optimize(lambda trial: _objective(trial, X_train, y_train), n_trials=500)
    best_model = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, **study.best_params)
    best_model.fit(X_train, y_train)
    y_test_pred_proba = best_model.predict_proba(X_test)
    y_test_pred = best_model.predict(X_test)
    logging.info("Best parameters found with logloss={:.4f} and accuracy={:.4f}: ".format(
        log_loss(y_test, y_test_pred_proba),
        accuracy_score(y_test, y_test_pred)
    ))
    for key, value in study.best_params.items():
        logging.info("{:<20} {:<9}".format(key, value))
    return best_model


def lgb_feature_importance(
        model: lgb.LGBMClassifier,
        feature_map: Dict[str, Dict[str, List[str]]],
        params: utils.HyperParams
) -> Dict[str, int]:
    """Evaluate relative feature importance according to LightGBM classifier and feature map, DEPRECATED"""
    importance = model.feature_importances_
    num_features = sum(
        len(feature_map[channel][field]) for channel in feature_map.keys() for field in feature_map[channel].keys()
    )
    acc_importance = [
        sum([importance[num_features * i + feature] for i in range(20*params.window_size)])
        for feature in range(num_features)
    ]
    feature_importance: Dict[str, int] = {}
    i = 0
    for channel in feature_map.keys():
        for field in feature_map[channel].keys():
            for element in feature_map[channel][field]:
                feature_importance["{:<5} {:<13} {:<27}".format(channel, field, element)] = acc_importance[i]
                i += 1
    for feature in feature_importance.keys():
        logging.info(feature + str(feature_importance[feature]))
    return feature_importance


if __name__ == "__main__":
    """Experiments of machine learning models for binary classification"""
    args = parser.parse_args()
    params = utils.HyperParams(json_path=os.path.join(args.experiment_dir, "params.json"))
    utils.set_logger(log_path=os.path.join(args.experiment_dir, "ml.log"))
    logging.info("Loading data...")
    X, y = get_data(params=params)
    logging.info("Comparing ML models...")
    models = model_selection(X, y, params=params)
    # logging.info("Evaluating LightGBM model feature importance...")
    # lgb_feature_importance(model=models["lgb"], feature_map=feature_map, params=params)
    # logging.info("Tuning hyperparameters for LightGBM...")
    # lgb_best_model = lgb_tuning(X, y, params=params)
    # logging.info("Evaluating fine-tuned LightGBM model feature importance...")
    # lgb_feature_importance(model=lgb_best_model, feature_map=feature_map, params=params)
