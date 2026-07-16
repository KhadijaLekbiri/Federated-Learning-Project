from model import NetflowClassifier
from client import Client
import numpy as np
import pandas as pd
import copy
import torch
from torch.utils.data import TensorDataset

# 1. Load raw dataframe
df = pd.read_parquet("NF-UNSW-NB15-V2.parquet")

num_clients = 5 

# 2. Separate features/labels, convert to tensors
y = df["Label"].values
X = df.drop(columns =["Label", "Attack"]).values.astype(np.float32)
X = (X - X.mean(axis=0))/ (X.std(axis=0)+1e-8)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

input_dim = X.shape[1]

