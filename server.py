

import copy
import torch

from evaluate import evaluate

class Server:
    def __init__(self, initial_model,lr=0.01,use_adam=False):
        self.model = initial_model
        self.optimizer = (
                    torch.optim.Adam(self.model.parameters(), lr=lr)
                    if use_adam else
                    torch.optim.SGD(self.model.parameters(), lr=lr)
                )
    
    def FedSGD_round(self, clients):
        
        averaged_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        for client in clients:
            client.receive_model(self.model.state_dict())

            gradients, loss = client._train_fedsgd()

            print("\tLoss:", loss)
            for name, grad in gradients.items():
                averaged_grads[name] += grad

        for name in averaged_grads:
            averaged_grads[name] /= len(clients)
        
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = averaged_grads[name]
        self.optimizer.step()

            
    def FedAVG_round(self, clients):
        
        averaged_weights = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        for client in clients:
            client.receive_model(self.model.state_dict())

            weights, loss = client._train_fedavg()

            print("\tLoss:", loss)
            for name, grad in weights.items():
                averaged_weights[name] += grad

        for name in averaged_weights:
            averaged_weights[name] /= len(clients)

        self.model.load_state_dict(averaged_weights)
        

    def Fed_train(self, clients, num_rounds, X_test, y_test, mode="fedsgd"):
        round_num = num_rounds
        while round_num:
            if mode == "fedsgd":
                self.FedSGD_round(clients)
            elif mode == "fedavg":
                self.FedAVG_round(clients)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            metrics = evaluate(self.model, X_test, y_test)
            print(f"Round {round_num}: {metrics}")
            print("************************")
            round_num -= 1
    