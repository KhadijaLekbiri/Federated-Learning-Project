

import copy
import torch
import numpy as np

from evaluate import evaluate
from client import clients_subset
from attacker import capture_leaks, score_reconstruction
from secure_agg import IdealSecureAggregation


class Server:
    def __init__(self, initial_model,data_size,lr=0.01,use_adam=False):
        self.model = initial_model
        self.optimizer = (
                    torch.optim.Adam(self.model.parameters(), lr=lr)
                    if use_adam else
                    torch.optim.SGD(self.model.parameters(), lr=lr)
                )
        self.data_size = data_size
    
    def _apply_gradient(self, gradient):
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = gradient[name]
        self.optimizer.step()

    def FedSGD_round(self, clients):
        
        averaged_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        # loss_history = {client.client_id: [] for client in clients}
        loss_history = {}

        for client in clients:
            client.receive_model(self.model.state_dict())

            gradients, loss = client._train_fedsgd()

            print("\tLoss:", loss)
            selected_data_size = sum(len(selected.dataset) for selected in clients)
            for name, grad in gradients.items():
                averaged_grads[name] += len(client.dataset) * grad / selected_data_size

            loss_history[client.client_id] = float(loss)

        # for name in averaged_grads:
        #     averaged_grads[name] /= len(clients)
        
        self._apply_gradient(averaged_grads)
        return loss_history

    def FedSGD_ideal_sa_round(self, clients):
        """Run FedSGD with an ideal-SA boundary around client gradients."""
        session = IdealSecureAggregation()
        loss_history = {}
        selected_data_size = sum(len(client.dataset) for client in clients)

        for client in clients:
            client.receive_model(self.model.state_dict())
            weight = len(client.dataset) / selected_data_size
            loss = client.train_fedsgd_secure(session, weight=weight)
            loss_history[client.client_id] = float(loss)

        aggregate_gradient = session.finalize()
        self._apply_gradient(aggregate_gradient)
        return loss_history
        

            
    def FedAVG_round(self, clients):
        
        averaged_weights = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        # loss_history = {client.client_id: [] for client in clients}
        loss_history = {}

        for client in clients:
            client.receive_model(self.model.state_dict())

            weights, loss = client._train_fedavg()

            print("\tLoss:", loss)
            for name, weight in weights.items():
                averaged_weights[name] += len(client.dataset)*weight/self.data_size
            
            loss_history[client.client_id] = float(loss)


        for name in averaged_weights:
            averaged_weights[name] /= len(clients)

        self.model.load_state_dict(averaged_weights)
        
        return loss_history

        

    def Fed_train(self, clients, num_rounds, X_test, y_test, mode="fedsgd"):
        history = {
            "round": [],
            "global_metrics": [],       
            "client_losses": [],      
        }
        
        rng = np.random.default_rng(seed=42)
        
        num_parts_list = rng.integers(
                low= int(0.4*len(clients)),
                high=len(clients) + 1,
                size=num_rounds
            ).tolist()
        
        leaked_records = []
        for i in range(num_rounds):
            num_part = num_parts_list[i]

            part_clts = clients_subset(clients, num_part, rng)
            init_theta = copy.deepcopy(self.model.state_dict())

            if mode == "fedsgd":
                round_losses = self.FedSGD_round(part_clts)
                leaked_records.extend(capture_leaks(part_clts, init_theta, i))

            elif mode == "ideal_sa":
                round_losses = self.FedSGD_ideal_sa_round(part_clts)

            elif mode == "fedavg":
                round_losses = self.FedAVG_round(part_clts)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            

            metrics = evaluate(self.model, X_test, y_test)
            history["round"].append(i)
            history["global_metrics"].append(metrics)
            history["client_losses"].append(round_losses)

            print(f"Round {i}: {metrics}")
            print("************************")

        return history, leaked_records
    
