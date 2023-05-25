import datetime
import json
from typing import Dict, List

import numpy as np

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

from preprocess import GNBLogFile


if __name__ == "__main__":
    # basic preprocess
    log = GNBLogFile(
        read_path="../data/NR/1st-example/gnb0.log",
        save_path="../data/NR/1st-example/export.json",
        timetable=[
            ((datetime.time(9, 48, 20), datetime.time(9, 58, 40)), "navigation_web"),
            ((datetime.time(10, 1, 40), datetime.time(10, 13, 20)), "streaming_youtube")
        ],
        window_size=1,
        tb_len_threshold=150
    )

    # read feature map from file
    with open("../experiments/base/features.json", 'r') as f:
        features: Dict[str, Dict[str, List[str]]] = json.load(f)

    # extract key_info vector
    for sample in log.samples:
        for record in sample:
            key_info: List[str or float or int] = []
            if record.basic_info["channel"] in features.keys():
                for feature in features[record.basic_info["channel"]]["basic_info"]:
                    if feature in record.basic_info.keys():
                        key_info.append(record.basic_info[feature])
                    else:
                        key_info.append(-1)
                for feature in features[record.basic_info["channel"]]["short_message"]:
                    if feature in record.short_message.keys():
                        key_info.append(record.short_message[feature])
                    else:
                        key_info.append(-1)
                for feature in features[record.basic_info["channel"]]["long_message"]:
                    if feature in record.long_message.keys():
                        key_info.append(record.long_message[feature])
                    else:
                        key_info.append(-1)
            record.key_info = key_info

    # process key_info vector to pure numeric
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

    # assemble combined vector for each sample
    sample_matrices = []
    for sample in log.samples:
        sample_matrix = []
        for subframe in range(10):
            for channel in features.keys():
                found = False
                for record in sample:
                    if not found and record.basic_info["channel"] == channel and int(
                            record.basic_info["subframe"]) % 10 == subframe:
                        sample_matrix.extend(record.key_info)
                        found = True
                if not found:
                    sample_matrix.extend([-1] * sum([len(value) for value in features[channel].values()]))
        sample_matrices.append(sample_matrix)

    # remove empty label and corresponding sample
    sample_labels = log.sample_labels
    X = []
    y = []
    for idx, sample_label in enumerate(sample_labels):
        if sample_label in ["navigation_web", "streaming_youtube"]:
            X.append(sample_matrices[idx])
            y.append(sample_labels[idx])
    X = np.array(X)
    y = np.array(y)
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    # evaluate models
    X_train, X_test, y_train, y_test = train_test_split(X, y)
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
        print("\n>> {:<3} {:.4f}".format(model_name, accuracy_score(y_test, y_test_pred)))
        print(confusion_matrix(y_test, y_test_pred))
        if model_name == "lgb":
            print("\n>> LGBM feature importance analysis")
            importance = model.feature_importances_
            acc_importance = [sum([importance[68*i+feature] for i in range(10)]) for feature in range(68)]
            i = 0
            for channel in features.keys():
                for field in features[channel].keys():
                    for element in features[channel][field]:
                        print("{:<5} {:<13} {:<26} {:>3}".format(channel, field, element, acc_importance[i]))
                        i += 1
