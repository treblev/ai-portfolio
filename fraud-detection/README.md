This folder hosts Fraud Detection model development using ML libraries.

## Setup

Create and activate a virtual environment:

```bash
cd fraud-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### macOS note (XGBoost)

If importing `xgboost` fails with a missing `libomp.dylib`, install OpenMP:

```bash
brew install libomp
```

Excluded features whose correlation with fraud was an artifact of simulation design rather than generalizable behavioral signal.

## Data

Download from Kaggle:
https://www.kaggle.com/c/ieee-fraud-detection/data

Place `train_transaction.csv` and `train_identity.csv` in this folder.