
import pandas as pd
import numpy as np
import copy

import torch.nn as nn
from torch.utils.data import DataLoader
from model import NetflowClassifier


class Client:
    def __init__(self, client_id, dataset, input_dim, batch_size=32, device="cpu"):
        self.client_id = client_id
        self.dataset =  dataset
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.dataloader =  DataLoader(dataset, shuffle=True, batch_size=batch_size)
        self.device = device
        self.last_gradient = None
        self.model = None

    
    def receive_model(self, global_state_dict):
        self.model = NetflowClassifier(self.input_dim).to(self.device)
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
        

    def local_train(self):

        if self.model is None:
            raise RuntimeError("Client has no model yet")
        
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()

        x, y = next(iter(self.dataloader))
        x, y = x.to(self.device), y.to(self.device)

        self.model.zero_grad()
        output = self.model(x)
        loss = loss_fn(output, y)
        loss.backward()

        gradients = {
            name: param.grad.clone().detach()
            for name, param in self.model.named_parameters()
        }

        self.last_gradient = gradients
        return gradients, loss.item()
    
    def get_update(self):
        return self.last_gradient
    



# df = pd.read_parquet("NF-UNSW-NB15-V2.parquet")
# print(df.head)
# client = Client(client_id=0, dataset=df)
# client.receive_model(model.statedict())