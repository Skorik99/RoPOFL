import torch
from collections import defaultdict, OrderedDict

from ..personalized.fedavg import PerFedAvg


class FedAvgGTS(PerFedAvg):
    def __init__(self, strategy, ckpt_path, server_test, warmup_rounds=1):
        super().__init__(strategy, ckpt_path, server_test)
        self.warmup_rounds = warmup_rounds

        # cluster_name -> aggregated model
        self.cluster_models = {}

        # cluster_name -> {client_id: weight}
        self.cluster_aggr_weights = {}

    def _init_federated(self, cfg):
        assert (
            "sharded" in cfg.distribution._target_
        ), "Now we support only sharded distribution"

        super()._init_federated(cfg)

        # Build all static information once
        self._build_client_sample_map()
        self._build_cluster_aggregation_weights()

        # Print aggregation weights once
        self._print_all_cluster_weights()

    def _build_client_sample_map(self):
        """
        Count number of samples per client from the dataset.
        """
        # Build per-client per-class counts once
        self.client_class_counts = defaultdict(lambda: defaultdict(int))
        for _, row in self.train_dataset.data.iterrows():
            client_id = int(row["client"])
            cls = int(row["target"])
            self.client_class_counts[client_id][cls] += 1

    def _build_cluster_aggregation_weights(self):
        """
        Compute aggregation weights for each cluster based ONLY on the number
        of samples belonging to the dominant classes of the target cluster.
        """

        distribution = self.train_dataset.distribution
        connected_clusters = distribution.connected
        num_dominants = self.cfg.distribution.num_dominants
        byzt_clients = set(getattr(self.server, "byzt_clients", set()))

        self.cluster_aggr_weights = {}

        for cluster_name, cluster_clients in self.strategy_map.items():
            # Parse cluster id from "Cluster_i"
            cluster_id = int(cluster_name.split("_")[1])

            # Dominant classes of THIS cluster
            dom_start = cluster_id * num_dominants
            dom_classes = set(range(dom_start, dom_start + num_dominants))

            # Clusters participating in aggregation
            all_cluster_ids = {cluster_id} | connected_clusters[cluster_id]

            # Collect all participating clients
            participating_clients = []
            for rank in all_cluster_ids:
                cname = f"Cluster_{rank}"
                participating_clients.extend(self.strategy_map[cname])

            non_byzt_clients = [
                cid for cid in participating_clients if cid not in byzt_clients
            ]
            if non_byzt_clients:
                weight_clients = non_byzt_clients
            else:
                weight_clients = participating_clients

            # Count ONLY samples of dominant classes
            client_dom_samples = {}
            for client_id in weight_clients:
                cnt = sum(
                    self.client_class_counts[client_id].get(c, 0) for c in dom_classes
                )
                client_dom_samples[client_id] = cnt

            total_dom_samples = sum(client_dom_samples.values())

            # Normalize into weights
            weights = {}
            for client_id in participating_clients:
                if non_byzt_clients and client_id in byzt_clients:
                    weights[client_id] = 0.0
                    continue
                cnt = client_dom_samples.get(client_id, 0)
                if total_dom_samples > 0:
                    weights[client_id] = cnt / total_dom_samples
                else:
                    weights[client_id] = 0.0

            self.cluster_aggr_weights[cluster_name] = weights

    def aggregate(self):
        """
        Aggregate client updates into one model per cluster.
        """
        if (
            self.warmup_rounds > 0
            and self.cur_round is not None
            and self.cur_round < self.warmup_rounds
        ):
            return self._aggregate_global_uniform()

        self.server.global_model.to("cpu")
        global_state = self.server.global_model.state_dict()

        participating = set(self.list_clients or [])

        for cluster_name, weights in self.cluster_aggr_weights.items():
            cluster_clients = self.strategy_map.get(cluster_name, [])
            active_clients = [c for c in cluster_clients if c in participating]
            if not active_clients:
                self.cluster_models[cluster_name] = self.cluster_models.get(
                    cluster_name, global_state
                )
                continue

            # Initialize empty delta
            cluster_delta = OrderedDict()
            for k in global_state.keys():
                cluster_delta[k] = torch.zeros_like(global_state[k], device="cpu")

            # Weighted aggregation of client gradients
            active_weights = {cid: weights.get(cid, 0.0) for cid in active_clients}
            weight_sum = sum(active_weights.values())
            if weight_sum <= 0:
                norm_weights = {
                    cid: 1.0 / len(active_clients) for cid in active_clients
                }
            else:
                norm_weights = {
                    cid: w / weight_sum for cid, w in active_weights.items()
                }

            for client_id, w in norm_weights.items():
                for name, grad in self.server.client_gradients[client_id].items():
                    cluster_delta[name] += (grad * w).to(cluster_delta[name].dtype)

            # Add delta to global model
            base_state = self.cluster_models.get(cluster_name, global_state)
            cluster_state = OrderedDict()
            for k in global_state.keys():
                cluster_state[k] = base_state[k] + cluster_delta[k]

            self.cluster_models[cluster_name] = cluster_state

        self.server.global_model.to(self.server.device)
        return global_state  # no changing in global model after aggregate

    def _aggregate_global_uniform(self):
        self.server.global_model.to("cpu")
        global_state = self.server.global_model.state_dict()
        participating = set(self.list_clients or [])
        byzt_clients = set(getattr(self.server, "byzt_clients", set()))
        non_byzt_clients = [c for c in participating if c not in byzt_clients]
        if not non_byzt_clients:
            self.server.global_model.to(self.server.device)
            return global_state

        updated_state = OrderedDict()
        weight = 1.0 / len(non_byzt_clients)
        for k in global_state.keys():
            updated_state[k] = global_state[k].clone()

        for client_id in non_byzt_clients:
            for name, grad in self.server.client_gradients[client_id].items():
                updated_state[name] += (grad * weight).to(updated_state[name].dtype)

        # Seed cluster models so the first post-warmup round can send updates.
        self.cluster_models = {
            cluster_name: updated_state for cluster_name in self.strategy_map.keys()
        }

        self.server.global_model.to(self.server.device)
        return updated_state

    def get_communication_content(self, rank):
        """
        Send cluster-specific model to the client.
        """

        content = super().get_communication_content(rank)
        if self.cur_round != 0 and (
            self.warmup_rounds <= 0 or self.cur_round >= self.warmup_rounds
        ):
            cluster_name = self.clients_strategy[rank]
            content["update_model"] = self.cluster_models.get(
                cluster_name, self.server.global_model.state_dict()
            )
        return content

    def train_round(self):
        if (
            self.warmup_rounds > 0
            and self.cur_round is not None
            and self.cur_round < self.warmup_rounds
        ):
            byzt_clients = set(getattr(self.server, "byzt_clients", set()))
            self.list_clients = [
                cid for cid in range(self.amount_of_clients) if cid not in byzt_clients
            ]
            self.list_clients.sort()
            self.server.list_clients = self.list_clients
        return super().train_round()

    def _print_all_cluster_weights(self):
        """
        Print aggregation weights for all clusters once.
        """
        print("\n" + "#" * 90)
        print("CLUSTER FEDAVG – AGGREGATION WEIGHTS (STATIC)")
        print("#" * 90)

        for cluster_name, weights in self.cluster_aggr_weights.items():
            cluster_id = int(cluster_name.split("_")[1])
            connected = self.train_dataset.distribution.connected[cluster_id]
            num_dominants = self.cfg.distribution.num_dominants
            dom_start = cluster_id * num_dominants
            dom_classes = set(range(dom_start, dom_start + num_dominants))

            print("\n" + "=" * 80)
            print(f"{cluster_name}")
            print(f"Connected clusters: {[cluster_id] + sorted(list(connected))}")
            print("-" * 80)
            print(
                f"{'Client':>6} | {'Client cluster':>14} | "
                f"{'#samples':>9} | {'weight':>10}"
            )
            print("-" * 80)

            for client_id, w in sorted(weights.items()):
                print(
                    f"{client_id:>6} | "
                    f"{self.clients_strategy[client_id]:>14} | "
                    f"{sum(self.client_class_counts[client_id].get(c, 0) for c in dom_classes):>9} | "
                    f"{w:>10.6f}"
                )

            print(f"Sum of weights: {sum(weights.values()):.6f}")
            print("=" * 80)

        print("#" * 90 + "\n")
