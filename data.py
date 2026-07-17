import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def load_and_preprocess(path="NF-UNSW-NB15-V2.parquet"):
    
    df = pd.read_parquet(path)

    y = df["Label"].values
    X = df.drop(columns=["Label", "Attack"]).values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    input_dim = X.shape[1]
    return X, y, input_dim


def make_train_test_split(X, y, test_size=0.15, seed=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def make_client_datasets(X_train, y_train, num_clients=5, seed=42):
    
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(len(X_train))
    client_indices = np.array_split(indices, num_clients)

    datasets = []
    for idx in client_indices:
        datasets.append(TensorDataset(X_train[idx], y_train[idx]))
    return datasets