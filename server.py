

import copy
import torch
import numpy as np

from evaluate import evaluate
from client import clients_subset


class Server:
    def __init__(self, initial_model,data_size,lr=0.01,use_adam=False):
        self.model = initial_model
        self.optimizer = (
                    torch.optim.Adam(self.model.parameters(), lr=lr)
                    if use_adam else
                    torch.optim.SGD(self.model.parameters(), lr=lr)
                )
        self.data_size = data_size
    
    def FedSGD_round(self, clients):
        
        averaged_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        loss_history = {client.client_id: [] for client in clients}

        for client in clients:
            client.receive_model(self.model.state_dict())

            gradients, loss = client._train_fedsgd()

            print("\tLoss:", loss)
            for name, grad in gradients.items():
                averaged_grads[name] += len(client.dataset)*grad/self.data_size

            loss_history[client.client_id].append(loss)

        for name in averaged_grads:
            averaged_grads[name] /= len(clients)
        
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = averaged_grads[name]
        self.optimizer.step()


        

            
    def FedAVG_round(self, clients):
        
        averaged_weights = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        loss_history = {client.client_id: [] for client in clients}

        for client in clients:
            client.receive_model(self.model.state_dict())

            weights, loss = client._train_fedavg()

            print("\tLoss:", loss)
            for name, weight in weights.items():
                averaged_weights[name] += len(client.dataset)*weight/self.data_size
            
            loss_history[client.client_id].append(loss)


        for name in averaged_weights:
            averaged_weights[name] /= len(clients)

        self.model.load_state_dict(averaged_weights)
        

    def Fed_train(self, clients, num_rounds, num_parts, X_test, y_test, mode="fedsgd"):
        rng = np.random.default_rng(seed=42)

        for i in range(num_rounds):
            part_clts = clients_subset(clients, num_parts, rng)
            if mode == "fedsgd":
                self.FedSGD_round(part_clts)
            elif mode == "fedavg":
                self.FedAVG_round(part_clts)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            metrics = evaluate(self.model, X_test, y_test)
            print(f"Round {i}: {metrics}")
            print("************************")
    