import argparse
import datetime
import json
import logging
import os
from typing import Dict, List

import numpy as np

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

import utils
from preprocess import GNBLogFile


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data/NR/1st-example")
parser.add_argument("--experiment_dir", default="../experiments/base")  # hyper-parameter json file


if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.HyperParams(os.path.join(args.experiment_dir, "params.json"))
    utils.set_logger("./ML.log")

    logging.info("Preprocessing...")
    log = GNBLogFile(
        read_path=os.path.join(args.data_dir, "gnb0.log"),
        feature_path=os.path.join(args.experiment_dir, "features.json"),
        save_path=os.path.join(args.data_dir, "export.json"),
        timetable=[
            ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
            ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
        ],
        window_size=params.window_size,
        tb_len_threshold=params.tb_len_threshold
    )

    logging.info("Processing key_info vector to pure numeric...")
    for sample in log.samples:
        for record in sample:
            for i in range(len(record.key_info)):
                try:
                    record.key_info[i] = eval(record.key_info[i])
                except (NameError, TypeError, SyntaxError) as _:
                    try:
                        record.key_info[i] = eval("".join([str(ord(c)) for c in record.key_info[i]]))
                    except TypeError as _:
                        pass

    logging.info("Assemble combined vector for each sample...")
    sample_matrices = []
    for sample in log.samples:
        sample_matrix = []
        for subframe in range(10):
            for channel in log.feature_map.keys():
                channel_in_subframe_flag = False
                for record in sample:
                    if (
                        not channel_in_subframe_flag and
                        record.basic_info["channel"] == channel and
                        int(record.basic_info["subframe"]) % 10 == subframe
                    ):
                        sample_matrix.extend(record.key_info)
                        channel_in_subframe_flag = True
                if not channel_in_subframe_flag:
                    sample_matrix.extend([-1] * sum([len(value) for value in log.feature_map[channel].values()]))
        sample_matrices.append(sample_matrix)

    logging.info("Remove empty label and corresponding sample...")
    sample_labels = log.sample_labels
    X_list: List[List[int or float]] = []
    y_list: List[str] = []
    for idx, sample_label in enumerate(sample_labels):
        if sample_label in ["navigation_web", "streaming_youtube"]:
            X_list.append(sample_matrices[idx])
            y_list.append(sample_labels[idx])
    X: np.ndarray = np.array(X_list)
    y: np.ndarray = np.array(y_list)
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    logging.info("Evaluate ML models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)
    models = {
        "sgd": SGDClassifier(),
        "svc": SVC(),
        "rf": RandomForestClassifier(),
        "mlp": MLPClassifier(),
        "tree": ExtraTreeClassifier(),
        "xgb": XGBClassifier(),
        "lgb": LGBMClassifier()
    }
    for model_name in models.keys():
        model = models[model_name]
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        logging.info("\n>> {:<3} {:.4f}".format(model_name, accuracy_score(y_test, y_test_pred)))
        logging.info(confusion_matrix(y_test, y_test_pred))
        if model_name == "lgb":
            logging.info("\n>> LGBM feature importance analysis")
            importance = model.feature_importances_
            acc_importance = [sum([importance[68*i+feature] for i in range(10)]) for feature in range(68)]
            i = 0
            for channel in log.feature_map.keys():
                for field in log.feature_map[channel].keys():
                    for element in log.feature_map[channel][field]:
                        logging.info("{:<5} {:<13} {:<26} {:>3}".format(channel, field, element, acc_importance[i]))
                        i += 1
