import copy
from types import MethodType

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from attacks.base.base_server import BaseAttackServer
from .pfedba_utils import (
    PATCH_SIZE,
    PFedBAPoisonedDataset,
    apply_patch_trigger,
    build_pfedba_trigger_loader,
    filter_non_target,
    get_key_column,
    get_normalized_bounds,
)


class PFedBAServer(BaseAttackServer):
    def __init__(
        self,
        byzantine_clients,
        attack_cfg,
        attack_client_cls,
        train_dataset,
        client_attack_map,
        cfg,
        is_personalized_regime=False,
    ):
        self.byzantine_clients = set(byzantine_clients)
        self.attack_cfg = attack_cfg
        self.attack_client_cls = attack_client_cls
        self.train_dataset = train_dataset
        self.client_attack_map = dict(client_attack_map)
        self.cfg = cfg
        self.is_personalized_regime = is_personalized_regime

    def set_client_result(self, client_result):
        self._pfedba_original_set_client_result(client_result)
        pfedba_eval = client_result.get("pfedba_eval")
        if pfedba_eval is None:
            return
        rank = client_result["rank"]
        server_asr = pfedba_eval.get("server_asr")
        if server_asr is not None:
            (
                self.pfedba_server_metrics[rank],
                self.pfedba_server_losses[rank],
                self.pfedba_server_df_len[rank],
            ) = server_asr
        client_asr = pfedba_eval.get("client_asr")
        if client_asr is not None:
            (
                self.pfedba_client_metrics[rank],
                self.pfedba_client_losses[rank],
                self.pfedba_client_df_len[rank],
            ) = client_asr

    def _aggregate_metric_triplets(
        self, metrics_list, losses_list, lens_list, participating_clients=None
    ):
        if participating_clients is None:
            participating_clients = range(self.amount_of_clients)
        server_metrics = []
        val_losses = []
        val_len_dfs = []
        for rank in participating_clients:
            if rank in self.pfedba_byzantine_clients:
                continue
            metrics = metrics_list[rank]
            loss = losses_list[rank]
            df_len = lens_list[rank]
            if metrics is None or loss is None or df_len == 0:
                continue
            server_metrics.append(metrics)
            val_losses.append(loss)
            val_len_dfs.append(df_len)

        if len(server_metrics) == 0:
            return None

        weights = [val_len_df / sum(val_len_dfs) for val_len_df in val_len_dfs]
        metrics_names = server_metrics[0].index
        if self.metric_aggregation == "uniform":
            val_loss = np.mean(val_losses)
            metrics = pd.concat(server_metrics).groupby(level=0).mean()
        else:
            val_loss = np.sum(
                [loss * weight for loss, weight in zip(val_losses, weights)]
            )
            metrics = sum(
                weight * metric for weight, metric in zip(weights, server_metrics)
            )
        metrics = metrics.reindex(metrics_names)
        return {
            "metrics": metrics,
            "loss": float(val_loss),
            "total": int(sum(val_len_dfs)),
        }

    def aggregate_pfedba_metrics(self, participating_clients=None, post_train=False):
        clean_metrics = self._aggregate_metric_triplets(
            self.clients_metrics if post_train else self.server_metrics,
            self.clients_losses if post_train else self.server_losses,
            self.server_val_df_len,
            participating_clients=participating_clients,
        )
        asr_metrics = self._aggregate_metric_triplets(
            self.pfedba_client_metrics if post_train else self.pfedba_server_metrics,
            self.pfedba_client_losses if post_train else self.pfedba_server_losses,
            self.pfedba_client_df_len if post_train else self.pfedba_server_df_len,
            participating_clients=participating_clients,
        )
        if clean_metrics is None or asr_metrics is None:
            return None
        return {
            "clean_metrics": clean_metrics["metrics"],
            "clean_loss": clean_metrics["loss"],
            "asr_metrics": asr_metrics["metrics"],
            "asr_loss": asr_metrics["loss"],
            "asr": float(asr_metrics["metrics"].loc["Accuracy"].iloc[0]),
        }

    def reset_pfedba_metrics(self):
        self.pfedba_server_metrics = [None for _ in range(self.amount_of_clients)]
        self.pfedba_server_losses = [None for _ in range(self.amount_of_clients)]
        self.pfedba_server_df_len = [0 for _ in range(self.amount_of_clients)]
        self.pfedba_client_metrics = [None for _ in range(self.amount_of_clients)]
        self.pfedba_client_losses = [None for _ in range(self.amount_of_clients)]
        self.pfedba_client_df_len = [0 for _ in range(self.amount_of_clients)]

    def should_run_pfedba_test(self):
        if hasattr(self, "server_test") and not self.server_test:
            return False
        return hasattr(self, "test_df") and hasattr(self, "test_loader")

    def get_pfedba_cluster_metrics(
        self,
        metrics_list,
        losses_list,
        participating_clients=None,
    ):
        if participating_clients is None:
            participating_clients = set(range(self.cfg.federated_params.amount_of_clients))
        else:
            participating_clients = set(participating_clients)

        cluster_metrics = {
            strategy: [
                metrics_list[i]
                for i in self.strategy_map[strategy]
                if (i not in self.byzt_clients and i in participating_clients)
                and metrics_list[i] is not None
            ]
            for strategy in self.strategy_map.keys()
        }
        cluster_losses = {
            strategy: [
                losses_list[i]
                for i in self.strategy_map[strategy]
                if (i not in self.byzt_clients and i in participating_clients)
                and losses_list[i] is not None
            ]
            for strategy in self.strategy_map.keys()
        }
        return cluster_metrics, cluster_losses

    def print_pfedba_cluster_metrics(self, cluster_metrics, cluster_losses, header):
        for strategy in self.strategy_map.keys():
            print(f"\n-------- {header} {strategy} cluster metrics --------")
            if not cluster_metrics[strategy]:
                print("No participating clients for this cluster.")
                continue
            metrics = pd.concat(cluster_metrics[strategy]).groupby(level=0).mean()
            loss = np.mean(cluster_losses[strategy])
            print(metrics)
            print(f"{header} Loss: {loss}")

    def test_pfedba_sharded_clusters(self):
        if self.test_sharded_df is None or self.test_sharded_cluster_to_clients is None:
            return
        if self.pfedba_delta is None:
            return

        print("\nServer PFedBA ASR Results by Sharded Clusters:")
        orig_test_loader = self.test_loader
        orig_test_df = self.test_df
        saved_global_state = copy.deepcopy(self.global_model.state_dict())
        delta = self.pfedba_delta.detach().cpu()
        target_label = int(self.pfedba_target_label)
        try:
            participating = set(self.list_clients or [])
            for name, client_ids in self.test_sharded_cluster_to_clients.items():
                cluster_clients = self.strategy_map.get(name, [])
                active_clients = [
                    c
                    for c in cluster_clients
                    if c in participating and c not in self.byzt_clients
                ]
                if not active_clients:
                    print(f"{name}: no participating benign clients, skipping.")
                    continue

                cluster_df = copy.deepcopy(self.test_sharded_df)
                cluster_df.data = self.test_sharded_df.data[
                    self.test_sharded_df.data["client"].isin(client_ids)
                ].reset_index(drop=True)
                cluster_df.mode = "test"
                cluster_df.data = filter_non_target(cluster_df.data, target_label)
                if len(cluster_df.data) == 0:
                    print(f"{name}: no eligible non-target samples, skipping.")
                    continue

                key_col = get_key_column(cluster_df.data)
                poisoned_keys = set(cluster_df.data[key_col].tolist())
                poisoned_dataset = PFedBAPoisonedDataset(
                    cluster_df,
                    delta_patch=delta,
                    target_label=target_label,
                    poisoned_keys=poisoned_keys,
                )
                self.test_loader = DataLoader(
                    poisoned_dataset,
                    batch_size=self.cfg.training_params.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.training_params.num_workers,
                    drop_last=False,
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
                print(f"{name} PFedBA ASR Loss: {mean_loss}")
                print(f"{name} PFedBA ASR: {float(mean_metrics.loc['Accuracy'].iloc[0])}")
        finally:
            self.global_model.load_state_dict(saved_global_state)
            self.test_loader = orig_test_loader
            self.test_df = orig_test_df

    def evaluate_pfedba_loader(self, eval_loader):
        self.global_model.to(self.device)
        self.global_model.eval()

        eval_loss = 0.0
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, (inputs, targets) in eval_loader:
                inp = inputs[0].to(self.device)
                targets = targets.to(self.device)
                outputs = self.global_model(inp)
                eval_loss += self.criterion(outputs, targets).detach().item()
                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        eval_loss = eval_loss / max(len(eval_loader), 1)
        metrics = self.model_trainer.calculate_metrics(
            fin_targets, fin_outputs, verbose=False
        )
        return metrics, eval_loss

    def evaluate_pfedba_test_metrics(self):
        if self.pfedba_delta is None:
            self.pfedba_test_metrics = None
            return None

        delta = self.pfedba_delta.detach().cpu()
        target_label = int(self.pfedba_target_label)
        test_df = filter_non_target(self.test_df.data, target_label)
        if len(test_df) == 0:
            self.pfedba_test_metrics = None
            return None

        clean_dataset = copy.deepcopy(self.test_df)
        clean_dataset.data = test_df
        clean_loader = DataLoader(
            clean_dataset,
            batch_size=self.cfg.training_params.batch_size,
            shuffle=False,
            num_workers=self.cfg.training_params.num_workers,
            drop_last=False,
        )
        clean_metrics, clean_loss = self.evaluate_pfedba_loader(clean_loader)

        eval_dataset = copy.deepcopy(clean_dataset)
        key_col = get_key_column(test_df)
        poisoned_keys = set(test_df[key_col].tolist())
        poisoned_dataset = PFedBAPoisonedDataset(
            eval_dataset,
            delta_patch=delta,
            target_label=target_label,
            poisoned_keys=poisoned_keys,
        )
        eval_loader = DataLoader(
            poisoned_dataset,
            batch_size=self.cfg.training_params.batch_size,
            shuffle=False,
            num_workers=self.cfg.training_params.num_workers,
            drop_last=False,
        )
        trigger_metrics, trigger_loss = self.evaluate_pfedba_loader(eval_loader)
        metrics = {
            "clean_metrics": clean_metrics,
            "clean_loss": clean_loss,
            "trigger_metrics": trigger_metrics,
            "trigger_loss": trigger_loss,
            "asr": float(trigger_metrics.loc["Accuracy"].iloc[0]),
        }
        self.pfedba_test_metrics = metrics
        return metrics

    def test_global_model(self, *args, **kwargs):
        self._pfedba_original_test_global_model(*args, **kwargs)
        if self.pfedba_is_personalized_regime:
            if self.cur_round != 0:
                server_cluster_metrics, server_cluster_losses = (
                    self.get_pfedba_cluster_metrics(
                        self.pfedba_server_metrics,
                        self.pfedba_server_losses,
                        self.list_clients,
                    )
                )
                print("\nPFedBA Server ASR Results by Clusters:")
                self.print_pfedba_cluster_metrics(
                    server_cluster_metrics,
                    server_cluster_losses,
                    "PFedBA Server ASR",
                )
                if self.cfg.federated_params.print_client_metrics:
                    client_cluster_metrics, client_cluster_losses = (
                        self.get_pfedba_cluster_metrics(
                            self.pfedba_client_metrics,
                            self.pfedba_client_losses,
                            self.list_clients,
                        )
                    )
                    print("\nPFedBA Client ASR Results by Clusters:")
                    self.print_pfedba_cluster_metrics(
                        client_cluster_metrics,
                        client_cluster_losses,
                        "PFedBA Client ASR",
                    )
                if self.should_run_pfedba_test():
                    self.test_pfedba_sharded_clusters()
                self.reset_pfedba_metrics()
            return
        metrics = self.evaluate_pfedba_test_metrics()
        if metrics is None:
            return
        if metrics["clean_metrics"] is not None:
            print("\nServer PFedBA Clean Test Results:")
            print(metrics["clean_metrics"])
            print(f"Server PFedBA Clean Test Loss: {metrics['clean_loss']}")
        if metrics["trigger_metrics"] is not None:
            print("\nServer PFedBA Attack Test Results:")
            print(metrics["trigger_metrics"])
            print(f"Server PFedBA Attack Test Loss: {metrics['trigger_loss']}")
            print(f"Server PFedBA ASR: {metrics['asr']}")

    def train_pfedba_trigger_steps(self, delta_patch, loader, optimizer, steps, mode):
        if loader is None or steps <= 0:
            return

        device = self.device
        model = self.global_model
        criterion = self.criterion.to(device)
        lower, upper = get_normalized_bounds(self.cfg, device)
        params = [param for param in model.parameters() if param.requires_grad]
        iterator = iter(loader)
        model.to(device)
        model.eval()
        for _ in range(steps):
            try:
                _, (inputs, targets) = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                _, (inputs, targets) = next(iterator)

            clean_inputs = inputs[0].to(device)
            clean_targets = targets.to(device)
            poisoned_targets = torch.full_like(clean_targets, self.pfedba_target_label)
            poisoned_inputs = apply_patch_trigger(clean_inputs, delta_patch)

            optimizer.zero_grad()
            if mode == "loss":
                outputs = model(poisoned_inputs)
                loss = criterion(outputs, poisoned_targets)
            else:
                poisoned_loss = criterion(model(poisoned_inputs), poisoned_targets)
                clean_loss = criterion(model(clean_inputs), clean_targets)
                poisoned_grads = torch.autograd.grad(
                    poisoned_loss, params, create_graph=True, allow_unused=False
                )
                clean_grads = torch.autograd.grad(
                    clean_loss, params, create_graph=False, allow_unused=False
                )
                loss = sum(
                    (poisoned_grad - clean_grad.detach()).pow(2).sum()
                    for poisoned_grad, clean_grad in zip(poisoned_grads, clean_grads)
                )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta_patch.clamp_(lower, upper)

    def train_pfedba_trigger(self):
        trigger_loader = build_pfedba_trigger_loader(self)
        assert trigger_loader is not None, "PFedBA trigger loader is empty."

        device = self.device
        if self.pfedba_delta is None:
            self.pfedba_delta = torch.zeros(3, PATCH_SIZE, PATCH_SIZE, device=device)
        else:
            self.pfedba_delta = self.pfedba_delta.to(device)

        delta_patch = torch.nn.Parameter(self.pfedba_delta.clone())
        optimizer = torch.optim.Adam(
            [delta_patch],
            lr=self.pfedba_cfg.trigger_lr,
            weight_decay=self.pfedba_cfg.trigger_weight_decay,
        )
        if self.cur_round == 0:
            self.train_pfedba_trigger_steps(
                delta_patch,
                trigger_loader,
                optimizer,
                self.pfedba_cfg.trigger_steps_loss_align,
                mode="loss",
            )
        self.train_pfedba_trigger_steps(
            delta_patch,
            trigger_loader,
            optimizer,
            self.pfedba_cfg.trigger_steps_grad_align,
            mode="grad",
        )
        self.pfedba_delta = delta_patch.detach().cpu()
        self.pfedba_payload = {
            "delta": self.pfedba_delta.clone(),
            "target_label": int(self.pfedba_target_label),
        }

    def change_functionality(self, server_instance):
        server_instance._pfedba_original_set_client_result = server_instance.set_client_result
        server_instance._pfedba_original_test_global_model = server_instance.test_global_model
        server_instance.pfedba_byzantine_clients = set(self.byzantine_clients)
        server_instance.pfedba_cfg = OmegaConf.create(
            OmegaConf.to_container(self.attack_cfg, resolve=True)
        )
        server_instance.pfedba_attack_client_cls = self.attack_client_cls
        server_instance.pfedba_train_dataset = self.train_dataset
        server_instance.pfedba_client_attack_map = dict(self.client_attack_map)
        server_instance.pfedba_target_label = int(self.attack_cfg.target_label)
        server_instance.pfedba_payload = None
        server_instance.pfedba_delta = None
        server_instance.pfedba_test_metrics = None
        server_instance.pfedba_is_personalized_regime = self.is_personalized_regime
        server_instance.reset_pfedba_metrics = MethodType(
            PFedBAServer.reset_pfedba_metrics, server_instance
        )
        server_instance.reset_pfedba_metrics()
        server_instance.set_client_result = MethodType(
            PFedBAServer.set_client_result, server_instance
        )
        server_instance.test_global_model = MethodType(
            PFedBAServer.test_global_model, server_instance
        )
        server_instance.aggregate_pfedba_metrics = MethodType(
            PFedBAServer.aggregate_pfedba_metrics, server_instance
        )
        server_instance.get_pfedba_cluster_metrics = MethodType(
            PFedBAServer.get_pfedba_cluster_metrics, server_instance
        )
        server_instance.print_pfedba_cluster_metrics = MethodType(
            PFedBAServer.print_pfedba_cluster_metrics, server_instance
        )
        server_instance.test_pfedba_sharded_clusters = MethodType(
            PFedBAServer.test_pfedba_sharded_clusters, server_instance
        )
        server_instance._aggregate_metric_triplets = MethodType(
            PFedBAServer._aggregate_metric_triplets, server_instance
        )
        server_instance.evaluate_pfedba_test_metrics = MethodType(
            PFedBAServer.evaluate_pfedba_test_metrics, server_instance
        )
        server_instance.should_run_pfedba_test = MethodType(
            PFedBAServer.should_run_pfedba_test, server_instance
        )
        server_instance.evaluate_pfedba_loader = MethodType(
            PFedBAServer.evaluate_pfedba_loader, server_instance
        )
        server_instance.train_pfedba_trigger_steps = MethodType(
            PFedBAServer.train_pfedba_trigger_steps, server_instance
        )
        server_instance.train_pfedba_trigger = MethodType(
            PFedBAServer.train_pfedba_trigger, server_instance
        )
        return server_instance
