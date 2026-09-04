import unittest

import torch
from torch.utils.data import TensorDataset

from client import Client
from model import NetflowClassifier
from secure_agg import IdealSecureAggregation, ideal_secure_aggregate
from server import Server


class IdealSecureAggregateTests(unittest.TestCase):
    def test_sum_matches_direct_mathematical_sum(self):
        first = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0])}
        second = {"weight": torch.tensor([4.0, 5.0]), "bias": torch.tensor([-1.0])}

        result = ideal_secure_aggregate([first, second])

        torch.testing.assert_close(result["weight"], first["weight"] + second["weight"])
        torch.testing.assert_close(result["bias"], first["bias"] + second["bias"])

    def test_weighted_sum(self):
        updates = [{"x": torch.tensor([2.0])}, {"x": torch.tensor([6.0])}]
        result = ideal_secure_aggregate(updates, weights=[0.25, 0.75])
        torch.testing.assert_close(result["x"], torch.tensor([5.0]))

    def test_rejects_incompatible_updates(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            ideal_secure_aggregate(
                [{"x": torch.zeros(2)}, {"x": torch.zeros(3)}]
            )

    def test_session_rejects_duplicate_client(self):
        session = IdealSecureAggregation()
        session.submit(7, {"x": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, "more than once"):
            session.submit(7, {"x": torch.ones(1)})


class SecureBoundaryTests(unittest.TestCase):
    def test_ideal_sa_server_sees_no_client_gradient_or_batch(self):
        torch.manual_seed(1)
        input_dim = 3
        clients = []
        for client_id in range(2):
            x = torch.randn(4, input_dim)
            y = torch.tensor([0, 1, 0, 1])
            clients.append(
                Client(client_id, TensorDataset(x, y), input_dim, batch_size=4)
            )

        server = Server(NetflowClassifier(input_dim), data_size=8, lr=0.01)
        losses = server.FedSGD_ideal_sa_round(clients)

        self.assertEqual(set(losses), {0, 1})
        for client in clients:
            self.assertIsNone(client.last_gradient)
            self.assertIsNone(client.last_batch)


if __name__ == "__main__":
    unittest.main()
