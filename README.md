# GCN-TULHOR MODEL

## Getting Start
```bash
conda create -n gcntul python=3.9
conda activate gcntul
pip install -r requirements.txt
```

## Dataset & Config
Update `main.py`

```python
# Config
DATA_PATH = "/path/to/a/ho-dataset.csv"
BATCH_SIZE = 128
EMB_SIZE = 256
HIDDEN_SIZE = 256
EPOCHS_MLM = 4
EPOCHS_CLASSIFY = 150
NUM_HEADS = 4
```

## Running the Model

```bash
python main.py 
# or for setting up the GPU
bash run.sh
```
