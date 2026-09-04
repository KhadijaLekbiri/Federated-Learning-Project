
import pandas as pd
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NetflowClassifier


class Client:
    def __init__(self, client_id, dataset, input_dim, local_lr=0.01,local_epochs=1, batch_size=32, device="cpu"):
        self.client_id = client_id
        self.dataset =  dataset
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.dataloader =  DataLoader(dataset, shuffle=True, batch_size=batch_size)
        self.device = device


        self.local_epochs = local_epochs
        self.local_lr = local_lr

        self.last_gradient = None
        self.last_weights = None
        self.last_batch = None
        self.model = None

    
    def receive_model(self, global_state_dict):
        self.model = NetflowClassifier(self.input_dim).to(self.device)
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
        
        
        
    def _batch_for_indices(self, sample_indices=None):
        if sample_indices is None:
            return next(iter(self.dataloader))
        rows = [self.dataset[int(index)] for index in sample_indices]
        if not rows:
            raise ValueError("sample_indices cannot be empty")
        return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])

    def _compute_fedsgd_update(self, sample_indices=None):

        self.model.train()
        loss_fn = nn.CrossEntropyLoss()

        x, y = self._batch_for_indices(sample_indices)
        x, y = x.to(self.device), y.to(self.device)
        self.model.zero_grad()
        output = self.model(x)
        loss = loss_fn(output, y)
        loss.backward()

        gradients = {
            name: param.grad.clone().detach()
            for name, param in self.model.named_parameters()
        }

        return gradients, loss.item(), (x.detach().clone(), y.detach().clone())

    def _train_fedsgd(self, sample_indices=None):
        """Plain FedSGD: expose the update for the baseline attacker."""
        gradients, loss, batch = self._compute_fedsgd_update(sample_indices)
        self.last_gradient = gradients
        self.last_batch = batch
        return gradients, loss

    def train_fedsgd_secure(self, aggregation_session, weight=1.0, sample_indices=None):
        """Submit privately and return metadata only, never an individual update."""
        self.last_gradient = None
        self.last_batch = None
        gradients, loss, _ = self._compute_fedsgd_update(sample_indices)
        aggregation_session.submit(self.client_id, gradients, weight=weight)
        return loss
    
    def _train_fedavg(self):

        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        
        total_loss, n_batches = 0.0, 0
        for _ in range(self.local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        weights = copy.deepcopy(self.model.state_dict())
        
        self.last_weights = weights
        return weights, avg_loss
    

def clients_subset(clients, num_participants,rng):

    indices = rng.permutation(len(clients))
    selected_indices = indices[:num_participants]
    return [clients[i] for i in selected_indices]
