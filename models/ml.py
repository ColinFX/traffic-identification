import argparse
import datetime
import logging
import os

import numpy as np

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

import utils
from preprocess import GNBDataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data/NR/1st-example")
parser.add_argument("--experiment_dir", default="../experiments/base")  # hyper-parameter json file


if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.HyperParams(os.path.join(args.experiment_dir, "params.json"))
    utils.set_logger("./ML.log")

    logging.info("Preprocessing...")
    dataset = GNBDataset(
        read_paths=["../data/NR/1st-example/gnb0.log"],
        save_paths=["../data/NR/1st-example/export.json"],
        feature_path="../experiments/base/features.json",
        timetables=[[
            ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
            ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
        ]],
        window_size=1,
        tb_len_threshold=150
    )

    logging.info("Evaluate ML models...")
    dataset.X = np.reshape(dataset.X, (dataset.X.shape[0], -1))
    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, random_state=17)
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
        logging.info(">> {:<3} {:.4f}".format(model_name, accuracy_score(y_test, y_test_pred)))
        logging.info(confusion_matrix(y_test, y_test_pred))
        if model_name == "lgb":
            logging.info("\n>> LGBM feature importance analysis")
            importance = model.feature_importances_
            acc_importance = [sum([importance[68*i+feature] for i in range(10)]) for feature in range(68)]
            i = 0
            for channel in dataset.feature_map.keys():
                for field in dataset.feature_map[channel].keys():
                    for element in dataset.feature_map[channel][field]:
                        logging.info("{:<5} {:<13} {:<26} {:>3}".format(channel, field, element, acc_importance[i]))
                        i += 1
