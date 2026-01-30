import copy
import torch
import torch.nn.functional as F
import pandas as pd

from ..personalized.fedavg import PerFedAvg
from ..fedamp.fedamp_client import FedAMPClient
from ..fedamp.fedamp_server import FedAMPServer


class FedAMP(PerFedAvg):
    def __init__(
        self,
        strategy,
        ckpt_path,
        server_test,
        proximity,
        scaling,
        self_value,
        warmup_rounds=1,
    ):
        super().__init__(strategy, ckpt_path, server_test)
        self.proximity = proximity
        self.scaling = scaling
        self.self_value = self_value
        self.warmup_rounds = warmup_rounds

    def _init_client_cls(self):
        if (
            self.cfg.federated_params.client_subset_size
            < self.cfg.federated_params.amount_of_clients
            and self.warmup_rounds <= 0
        ):
            raise ValueError(
                "FedAMP partial participation requires warmup_rounds > 0 to "
                "initialize all client gradients."
            )
        super()._init_client_cls()
        self.client_cls = FedAMPClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.proximity])

    def _init_server(self, cfg):
        byzt_clients = {
            idx
            for idx, attack_type in self.client_attack_map.items()
            if attack_type != "no_attack"
        }
        self.server = FedAMPServer(cfg, self.server_test, byzt_clients=byzt_clients)

    def train_round(self):
        if (
            self.warmup_rounds > 0
            and self.cur_round is not None
            and self.cur_round < self.warmup_rounds
        ):
            self.list_clients = list(range(self.amount_of_clients))
            self.list_clients.sort()
            self.server.list_clients = self.list_clients
        return super().train_round()

    def define_aggregation_weights(self):
        with torch.no_grad():
            clients_params_vectors = []
            for client_rank in range(self.cfg.federated_params.amount_of_clients):
                personalized_model = self.server.client_gradients[client_rank]
                if not personalized_model:
                    raise ValueError(
                        "Missing client gradients for FedAMP similarity; "
                        "ensure warmup_rounds covers all clients."
                    )
                parameter_list = [
                    param.detach().cpu().flatten()
                    for param in personalized_model.values()
                ]
                vec = torch.cat(parameter_list)
                clients_params_vectors.append(vec)

            # Keep similarity computation on CPU to reduce GPU memory pressure
            clients_params_vectors = torch.stack(clients_params_vectors).float()

            # Cosine similarity via normalized dot products (stays on device)
            normed = F.normalize(clients_params_vectors, p=2, dim=1, eps=1e-12)
            sim_matrix = normed @ normed.T
            sim_matrix = sim_matrix / self.scaling

            aggregation_weights = self.softmax(sim_matrix)

            # Pretty-print aggregation weights as markdown table (clients are 1-indexed)
            df = pd.DataFrame(
                aggregation_weights.detach().cpu().numpy(),
                index=[
                    f"Client {i}" for i in range(1, aggregation_weights.shape[0] + 1)
                ],
                columns=[
                    f"Client {i}" for i in range(1, aggregation_weights.shape[1] + 1)
                ],
            )
            print("\nAggregation weights markdown table:\n")
            print(df.to_markdown())

            return aggregation_weights

    def aggregate(self):
        aggr_weights = self.define_aggregation_weights()
        global_model_state_dict = self.server.global_model.state_dict()
        n_clients = self.cfg.federated_params.amount_of_clients

        # Pre-calculation of weighted relative models for each client
        with torch.no_grad():
            self.server.relative_models = [{} for _ in range(n_clients)]
            # Keep aggregation math on CPU to avoid extra GPU memory use
            aggr_weights_tensor = aggr_weights.to(dtype=torch.float32)
            for key in global_model_state_dict.keys():
                grad_stack = torch.stack(
                    [
                        self.server.client_gradients[i][key].detach().cpu()
                        for i in range(n_clients)
                    ]
                )
                if not grad_stack.is_floating_point():
                    grad_stack = grad_stack.float()
                weighted = torch.einsum(
                    "ci,i...->c...",
                    aggr_weights_tensor.to(dtype=grad_stack.dtype),
                    grad_stack,
                )
                for client_rank in range(n_clients):
                    self.server.relative_models[client_rank][key] = weighted[
                        client_rank
                    ]

        return global_model_state_dict

    def get_communication_content(self, rank):
        # Don`t support finetuning by default
        content = {
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
            "strategy": self.strategy.get_client_payload(rank),
            # Send the same client model back, expect first round
            "update_model": {
                k: v.cpu()
                for k, v in (
                    self.server.client_gradients[rank].items()
                    if self.cur_round != 0
                    else self.server.global_model.state_dict().items()
                )
            },
            # Also send an aggregated model relative to other clients, expect first round
            "relative_model": {
                k: v.cpu()
                for k, v in (
                    self.server.relative_models[rank].items()
                    if self.cur_round != 0
                    else self.server.global_model.state_dict().items()
                )
            },
        }

        return content

    def softmax(self, x, axis=1):
        if self.self_value is None:
            return torch.softmax(x, dim=axis)
        else:
            # Weighted softmax with selfvalue
            x = x.clone()
            x.fill_diagonal_(-float("inf"))
            off_diag_weights = torch.softmax(x, dim=axis) * (1 - self.self_value)
            diag = torch.full(
                (x.size(0),),
                self.self_value,
                device=x.device,
                dtype=x.dtype,
            )
            weights = off_diag_weights
            weights.fill_diagonal_(self.self_value)
            return weights
