# traffic-identification
Research project on the identification of user traffic profiles in 5G (NSA and SA). 

## SrsRAN-LTE Quick Start

### Log file preprocessing

Let's say we have a folder `./data` containing all `.log` files each with their file name as the label for 
classification. 

* Step 1: Preprocess `.log` files and temporarily store them as `.pkl` files under the same folder:

```python
from preprocess import *

for file_path in t := tqdm.tqdm(utils.listdir_with_suffix("./data", ".log")):
    t.set_postfix({"step": "convert_log", "read_path": file_path})
    label = os.path.splitext(os.path.split(file_path)[1])[0]
    logfile = SrsRANLteLogFile(
        read_path=file_path,
        label=label,
        window_size=1,
        tbs_threshold=0
    )
    with open(os.path.join("./data", label+".pkl"), "wb") as file:
        pickle.dump(logfile, file)
```

* Step 2: Embed features of records in log files to numerical vectors: 

* Step 2.1: Count what features each physical channel has and collect datatype metadata of each column: 

```python
from dataloader import *

# collect
hybrid_encoder = SrsRANLteHybridEncoder()
pkl_file_paths = utils.listdir_with_suffix("./data", ".pkl")
hybrid_encoder.collect_columns_metadata(pkl_file_paths)
# save
hybrid_encoder.save_columns_metadata("./data/columns_metadata.json")
```

* Step 2.2: Fit either `MinMaxScaler` or `OneHotEncoder` for each feature based on the collected metadata:

```python
from dataloader import *

# load
hybrid_encoder = SrsRANLteHybridEncoder()
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
