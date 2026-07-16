

import copy
import torch

class Server:
    def __init__(self, initial_model,lr=0.01,use_adam=False):
        self.model = initial_model
        self.optimizer = (
                    torch.optim.Adam(self.model.parameters(), lr=lr)
                    if use_adam else
                    torch.optim.SGD(self.model.parameters(), lr=lr)
                )
    
    def FL_round(self, clients):
        
        averaged_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        for client in clients:
            client.receive_model(self.model.state_dict())

            gradients, loss = client.local_train()

            print("\tLoss:", loss)
            for name, grad in gradients.items():
                averaged_grads[name] += grad

        for name in averaged_grads:
            averaged_grads[name] /= len(clients)
        
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = averaged_grads[name]
        self.optimizer.step()

    def FL_algo(self, clients, num_rounds):
        c = num_rounds
        while c:
            self.FL_round(clients)
            print("************************")
            c -= 1
            

        