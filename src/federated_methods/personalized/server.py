import copy
from hydra.utils import instantiate
from ..fedavg.fedavg_server import FedAvgServer
import pandas as pd
import numpy as np
from collections import OrderedDict
from utils.data_utils import get_dataset_loader, print_df_distribution


class PerServer(FedAvgServer):
    def __init__(self, cfg, server_test, byzt_clients=None):
        super().__init__(cfg)
        self.server_test = server_test
        self.byzt_clients = set(byzt_clients or [])
        self.prev_global_model_state = None
        self._init_test_sharded_split()

    def _init_test_sharded_split(self):
        self.test_sharded_df = None
        self.test_sharded_cluster_to_clients = None
        if not self.server_test:
            return
        if "ShardedDistribution" not in str(self.cfg.distribution._target_):
            return
        if isinstance(self.test_df.data.iloc[0]["target"], list):
            return

        sharded_dist = instantiate(self.cfg.distribution)
        if hasattr(sharded_dist, "verbose"):
            sharded_dist.verbose = False

        test_df = copy.deepcopy(self.test_df)
        test_df.data = sharded_dist.split_to_clients(
            test_df.data,
            amount_of_clients=sharded_dist.n_clusters,
            random_state=self.cfg.random_state,
        )
        print_df_distribution(
            test_df.data,
            num_classes=test_df.num_classes,
            num_clients=sharded_dist.n_clusters,
        )

        cluster_to_clients = {
            f"Cluster_{i}": [i] for i in range(sharded_dist.n_clusters)
        }

        self.test_sharded_df = test_df
        self.test_sharded_cluster_to_clients = cluster_to_clients

    def test_global_model(self, dataset="test", require_metrics=True):
        if self.cur_round != 0:
            # Print metrics for global model over clusters
            print("\nServer Validation Results by Clusters:")
            server_cluster_metrics, server_cluster_losses = self.get_cluster_metrics(
                self.server_metrics, self.server_losses, self.list_clients
            )
            self.print_meaned_metrics(server_cluster_metrics, server_cluster_losses)
            # Print metrics for local models over clusters
            if self.cfg.federated_params.print_client_metrics:
                client_cluster_metrics, client_cluster_losses = (
                    self.get_cluster_metrics(
                        self.clients_metrics, self.clients_losses, self.list_clients
                    )
                )
                print("\n-------- MEANED METRICS AFTER FINETUNING --------")
                self.print_meaned_metrics(client_cluster_metrics, client_cluster_losses)
            # Print metrics for global model on test dataset
            if self.server_test:
                self.test_sharded_clusters()
            self.cleanup()
        else:
            self.global_model.to(self.device)

    def save_best_model(self, round):
        if (
            self.list_clients is not None
            and len(self.list_clients) != self.amount_of_clients
        ):
            self.checkpoint_path = None
            return
        super().save_best_model(round)

    def get_cluster_metrics(
        self, metrics_list, losses_list, participating_clients=None
    ):
        if participating_clients is None:
            participating_clients = set(
                range(self.cfg.federated_params.amount_of_clients)
            )
        else:
            participating_clients = set(participating_clients)

        cluster_metrics = {
            strategy: [
                metrics_list[i]
                for i in self.strategy_map[strategy]
                if (i not in self.byzt_clients and i in participating_clients)
            ]
            for strategy in self.strategy_map.keys()
        }
        cluster_losses = {
            strategy: [
                losses_list[i]
                for i in self.strategy_map[strategy]
                if (i not in self.byzt_clients and i in participating_clients)
            ]
            for strategy in self.strategy_map.keys()
        }
        return cluster_metrics, cluster_losses

    def print_meaned_metrics(self, cluster_metrics, cluster_losses):
        for strategy in self.strategy_map.keys():
            print(f"\n-------- Mean {strategy} cluster metrics --------")
            if not cluster_metrics[strategy]:
                print("No participating clients for this cluster.")
                continue
            metrics = pd.concat(cluster_metrics[strategy]).groupby(level=0).mean()
            loss = np.mean(cluster_losses[strategy])
            print(f"\nServer Valid Results:\n{metrics}")
            print(f"Server Valid Loss: {loss}")

    def test_sharded_clusters(self):
        if self.test_sharded_df is None or self.test_sharded_cluster_to_clients is None:
            return

        print("\nServer Test Results by Sharded Clusters:")
        orig_test_loader = self.test_loader
        orig_test_df = self.test_df
        saved_global_state = copy.deepcopy(self.global_model.state_dict())
        try:
            participating = set(self.list_clients or [])
            for name, client_ids in self.test_sharded_cluster_to_clients.items():
                cluster_clients = self.strategy_map.get(name, [])
                active_clients = [c for c in cluster_clients if c in participating]
                if not active_clients:
                    print(f"{name}: no participating clients, skipping.")
                    continue

                cluster_df = copy.deepcopy(self.test_sharded_df)
                cluster_df.data = self.test_sharded_df.data[
                    self.test_sharded_df.data["client"].isin(client_ids)
                ].reset_index(drop=True)
                cluster_df.mode = "test"
                self.test_loader = get_dataset_loader(
                    cluster_df, self.cfg, drop_last=False
                )
                cluster_metrics = []
                cluster_losses = []
                for client_rank in active_clients:
                    local_state = self._get_client_model_state(client_rank)
                    if local_state is None:
                        continue
                    self.global_model.load_state_dict(local_state)
                    metrics, loss = self._silent_test_metrics()
                    cluster_metrics.append(metrics)
                    cluster_losses.append(loss)

                if not cluster_metrics:
                    print(f"{name}: no valid client metrics, skipping.")
                    continue

                mean_metrics = pd.concat(cluster_metrics).groupby(level=0).mean()
                mean_loss = float(np.mean(cluster_losses))
                print(f"\n{name} client_ids: {active_clients}")
                print(mean_metrics)
                print(f"{name} Test Loss: {mean_loss}")
        finally:
            self.global_model.load_state_dict(saved_global_state)
            self.test_loader = orig_test_loader
            self.test_df = orig_test_df

    def _silent_test_metrics(self):
        fin_targets, fin_outputs, test_loss = self.model_trainer.server_eval_fn(self)
        metrics = self.model_trainer.calculate_metrics(
            fin_targets, fin_outputs, verbose=False
        )
        test_loss = test_loss.detach().cpu().item()
        return metrics, test_loss

    def _get_client_model_state(self, client_rank):
        if self.prev_global_model_state is None:
            return None
        base_state = self.prev_global_model_state
        grad_state = self.client_gradients[client_rank]
        return {
            k: base_state[k] + grad_state[k]
            for k in base_state.keys()
            if k in grad_state
        }

    def cleanup(self):
        # Emptying memory forcibly
        self.client_gradients = [
            OrderedDict() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        self.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
