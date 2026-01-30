from ..personalized.fedavg import PerFedAvg
from .federated_clustering_client import FederatedClusteringClient
from .federated_clustering_server import FederatedClusteringServer


class FederatedClustering(PerFedAvg):
    def __init__(
        self,
        strategy,
        ckpt_path,
        server_test,
        K,
        tau_percentile,
        clustering_iters,
        alpha,
        num_local_iters,
        eta,
        debug,
        warmup_rounds=1,
    ):
        super().__init__(strategy, ckpt_path, server_test)
        self.K = K
        self.tau_percentile = tau_percentile
        self.clustering_iters = clustering_iters
        self.alpha = alpha
        self.num_local_iters = num_local_iters
        self.eta = eta
        self.warmup_rounds = warmup_rounds
        self.debug = debug
        assert self.K > 0, f"K must be positive, got {self.K}"
        assert (
            0 < self.tau_percentile < 1.0
        ), f"tau_percentile must be in (0,1), got {self.tau_percentile}"
        assert (
            self.clustering_iters > 0
        ), f"clustering_iters must be positive, got {self.clustering_iters}"
        assert (
            self.num_local_iters > 0
        ), f"num_local_iters must be positive, got {self.num_local_iters}"
        assert self.eta > 0, f"eta must be positive, got {self.eta}"
        assert 0 < self.alpha <= 1.0, f"alpha must be in (0,1], got {self.alpha}"

    def _init_client_cls(self):
        if (
            self.cfg.federated_params.client_subset_size
            < self.cfg.federated_params.amount_of_clients
            and self.warmup_rounds <= 0
        ):
            raise ValueError(
                "FederatedClustering partial participation requires warmup_rounds > 0 "
                "to initialize all client gradients."
            )
        super()._init_client_cls()
        self.client_cls = FederatedClusteringClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.alpha, self.num_local_iters])

    def _init_server(self, cfg):
        # We do not test global model on server by default
        self.server_test = False
        byzt_clients = {
            idx
            for idx, attack_type in self.client_attack_map.items()
            if attack_type != "no_attack"
        }
        self.server = FederatedClusteringServer(
            cfg,
            self.server_test,
            self.K,
            self.tau_percentile,
            self.clustering_iters,
            self.eta,
            self.debug,
            byzt_clients=byzt_clients,
        )

    def aggregate(self):
        # Cluster momentums and update per-client models
        self.server.cluster_and_update_models()
        return self.server.global_model.state_dict()

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

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        if len(self.server.client_models[rank]) == 0:
            # First round, no client model to send to Client
            self.server.client_models[rank] = {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            }
        content["client_model"] = self.server.client_models[rank]
        # We remove update model as we do not use global model in Federated-Clustering
        content.pop("update_model", None)
        return content
