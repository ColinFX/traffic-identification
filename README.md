# traffic-identification
Research project on the identification of user traffic profiles in 5G (NSA and SA). 

## SrsRAN-LTE Quick Start

### Log file preprocessing

Let's say we have a folder `./data` containing all `.log` files each with their file name as the label for 
classification. 

* Step 1: Preprocess `.log` files and temporarily store them as `.pkl` files under the same folder:

```python
import os
from preprocess import *

for file_path in t := tqdm.tqdm(utils.listdir_with_suffix("./data", ".log")):
    t.set_postfix({"step": "convert_log", "read_path": file_path})
    label = os.path.splitext(os.path.split(file_path)[1])[0]
    logfile = SrsranLogFile(
        read_path=file_path,
        label=label,
        window_size=1,
        tbs_threshold=0
    )
    with open(os.path.join("./data", label + ".pkl"), "wb") as file:
        pickle.dump(logfile, file)
```

* Step 2: Embed features of records in log files to numerical vectors: 

* Step 2.1: Count what features each physical channel has and collect datatype metadata of each column:

```python
from dataloader import *

# collect
hybrid_encoder = HybridEncoder()
pkl_file_paths = utils.listdir_with_suffix("./data", ".pkl")
hybrid_encoder.collect_columns_metadata(pkl_file_paths)
# save
hybrid_encoder.save_columns_metadata("./data/columns_metadata.json")
```

* Step 2.2: Fit either `MinMaxScaler` or `OneHotEncoder` for each feature based on the collected metadata:

```python
from dataloader import *

# load
hybrid_encoder = HybridEncoder()
hybrid_encoder.load_columns_metadata("./data/columns_metadata.json")
# fit
pkl_file_paths = utils.listdir_with_suffix("./data", ".pkl")
hybrid_encoder.fit(pkl_file_paths)
# save
with open("./data/hybrid_encoder.pickle", "wb") as file:
    pickle.dump(hybrid_encoder, file)
```

* Step 2.3: Transform all features from different channels by assigning `embedded_message` of each record and then 
`X` of each sample: 

```python
from dataloader import *

# load
with open("./data/hybrid_encoder.pickle", "rb") as file:
    hybrid_encoder = pickle.load(file)
# transform
pkl_file_paths = utils.listdir_with_suffix("./data", ".pkl")
hybrid_encoder.transform(pkl_file_paths)
```

* Step 3: Train and evaluate Light Gradient Boosting Machine on this dataset:

```python
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataloader import *

# load
dataloaders = SrsranDataLoaders(
    params=utils.HyperParams("experiments/base/params.json"),
    split_percentages=[0.7, 0, 0.3],
    read_train_val_test_npz_paths=utils.listdir_with_suffix("./data", ".npz")
)
X_train = dataloaders.train_dataset.X.reshape(dataloaders.train_dataset.X.shape[0], -1)
y_train = dataloaders.train_dataset.y
X_test = dataloaders.test_dataset.X.reshape(dataloaders.test_dataset.X.shape[0], -1)
y_test = dataloaders.test_dataset.y

# train
model = lgb.LGBMClassifier(random_state=17)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(
    accuracy_score(y_true=y_test, y_pred=y_test_pred),
    precision_score(y_true=y_test, y_pred=y_test_pred, average="macro"),
    recall_score(y_true=y_test, y_pred=y_test_pred, average="macro"),
    f1_score(y_true=y_test, y_pred=y_test_pred, average="macro")
)
```

* Extra: change threshold and window size, redo Step 2.3 and then Step 3 after modification

```python
from preprocess import *

t = tqdm.tqdm(utils.listdir_with_suffix("./data", ".pkl"))
for file_path in t:
    t.set_postfix({"step": "change_th", "read_path": file_path})
    with open(file_path, "rb") as file:
        logfile: SrsranLogFile = pickle.load(file)
    logfile.samples = logfile.regroup_records(window_size=2)
    logfile.filter_samples(threshold=10)
    with open(file_path + "_2_10", "wb") as file2:
        pickle.dump(logfile, file2)
```


