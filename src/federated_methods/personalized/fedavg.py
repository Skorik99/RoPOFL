import torch
from collections import OrderedDict

from ..fedavg.fedavg import FedAvg
from .client import PerClient
from .server import PerServer
from .strategy import *
from .attack_utils import map_fixed_attack_clients


class PerFedAvg(FedAvg):
    def __init__(self, strategy, ckpt_path, server_test):
        self.strategy = strategy
        self.ckpt_path = ckpt_path
        self.server_test = server_test
        super().__init__()

    def _init_client_cls(self):
        if (
            self.cfg.federated_params.client_subset_size
            < self.cfg.federated_params.amount_of_clients
        ):
            assert (
                self.strategy == "sharded"
            ), "Partial participation for personalized methods is supported only with sharded strategy."
            print(
                "Partial participation enabled: personalized save_best_model is disabled."
            )

        super()._init_client_cls()
        self.client_cls = PerClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.define_clusters()

    def _init_server(self, cfg):
        byzt_clients = {
            idx
            for idx, attack_type in self.client_attack_map.items()
            if attack_type != "no_attack"
        }
        self.server = PerServer(cfg, self.server_test, byzt_clients=byzt_clients)

    def attack_setup(self, cfg):
        super().attack_setup(cfg)
        fixed = getattr(cfg.federated_params, "list_byzantines", None)
        if fixed:
            print(f"Using fixed byzantine list, swapping client_attack_map to: {fixed}")
            attack_type = (
                cfg.federated_params.clients_attack_types[0]
                if isinstance(cfg.federated_params.clients_attack_types, (list, tuple))
                else cfg.federated_params.clients_attack_types
            )
            self.client_attack_map = map_fixed_attack_clients(
                fixed, cfg.federated_params.amount_of_clients, attack_type
            )

    def define_clusters(self):
        self.num_clients = self.cfg.federated_params.amount_of_clients
        match self.strategy:
            case "sharded":
                self.strategy = ShardedStrategy()

            case "base":
                self.strategy = BaseStrategy()

            case "filter":
                self.strategy = FilterStrategy()

            case _:
                raise ValueError(f"No such cluster split type {self.strategy}")

        self.strategy_map, self.clients_strategy = self.strategy.split_clients(self)
        self.server.strategy_map = self.strategy_map
        print(f"Cluster mapping: {self.strategy_map}")

    def load_checkpoint(self):
        weights = torch.load(
            self.ckpt_path, map_location=self.server.device, weights_only=False
        )["model"]
        self.server.global_model.load_state_dict(weights)

    def get_communication_content(self, rank):
        # Fine-tune option
        if self.cur_round == 0 and self.ckpt_path is not None:
            self.load_checkpoint()

        # In we need additionaly send client cluster strategy
        content = super().get_communication_content(rank)
        content["strategy"] = self.strategy.get_client_payload(rank)
        return content

    def aggregate(self):
        if self.server_test:
            self.server.prev_global_model_state = {
                k: v.detach().cpu()
                for k, v in self.server.global_model.state_dict().items()
            }
        return super().aggregate()

    def log_round(self):
        # TODO: write log round for personalization strategies
        pass

    def cleanup(self):
        # We clenup memory after test_global_model
        pass
