import copy
import time

import torch

from ..personalized.fedavg import PerFedAvg
from .ropo_client import RoPOClient, ropo_multiprocess_client
from .ropo_server import RoPOServer
from .utils import aggregated_client_models_in_prev_round
from utils.attack_utils import apply_synchronized_attack
from utils.utils import read_runtime_caching


class RoPO(PerFedAvg):
    def __init__(
        self,
        strategy,
        ckpt_path,
        server_test,
        beta,
        C,
        theta,
        num_local_iters,
        theta_decay,
        global_decay,
        sgd_correction,
        num_steps_to_agg,
        start_steps_to_agg,
        warmup_rounds=1,
        similarity_by_head=False,
        freeze_vit_head=None,
        print_trial_scores=False,
    ):
        super().__init__(strategy, ckpt_path, server_test)
        self.beta = beta
        self.theta = theta
        self.num_local_iters = num_local_iters
        self.theta_decay = theta_decay
        self.global_decay = global_decay
        self.sgd_correction = sgd_correction
        self.C = C
        self.num_steps_to_agg = num_steps_to_agg
        self.start_steps_to_agg = start_steps_to_agg
        self.warmup_rounds = warmup_rounds
        self.similarity_by_head = similarity_by_head
        self.freeze_vit_head = freeze_vit_head
        self.print_trial_scores = print_trial_scores
        self._window = {"enabled": False, "round": None, "threshold": None}
        if self.num_steps_to_agg is not None:
            assert self.num_steps_to_agg > 0
            assert self.start_steps_to_agg is not None
            assert self.start_steps_to_agg >= 0

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = RoPOClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_kwargs["worker_target"] = ropo_multiprocess_client
        self.client_args.extend(
            [
                self.theta,
                self.num_local_iters,
                self.theta_decay,
                self.global_decay,
                self.sgd_correction,
                self.global_decay,
            ]
        )

    def _init_server(self, cfg):
        self.server_test = False
        byzt_clients = {
            idx
            for idx, attack_type in self.client_attack_map.items()
            if attack_type != "no_attack"
        }
        self.server = RoPOServer(
            cfg,
            self.server_test,
            self.beta,
            self.C,
            similarity_by_head=self.similarity_by_head,
            freeze_vit_head=self.freeze_vit_head,
            print_trial_scores=self.print_trial_scores,
            byzt_clients=byzt_clients,
        )

    def serialize_communication_content(self):
        self._window = read_runtime_caching(self)
        if self.cur_round == 0:
            return super().serialize_communication_content()
        return None

    def train_round(self):
        if (
            self.warmup_rounds > 0
            and self.cur_round is not None
            and self.cur_round < self.warmup_rounds
        ):
            self.list_clients = list(range(self.amount_of_clients))
            self.list_clients.sort()
            self.server.list_clients = self.list_clients

        self.clients_loader = self.manager.create_batches(self.list_clients)
        self.serialized_content = self.serialize_communication_content()

        for clients_batch in self.clients_loader:
            self.manager.set_ranks_to_procs(clients_batch)

            for pipe_num, rank in clients_batch:
                content = self.get_communication_content(rank)
                self.server.send_content_to_client(pipe_num, content)

            for pipe_num, rank in clients_batch:
                content = self.server.rcv_content_from_client(pipe_num)
                self.parse_communication_content(copy.deepcopy(content))

        self.server.client_gradients = apply_synchronized_attack(
            self.list_clients,
            self.server.client_gradients,
            self.client_map_round,
            self.attack_configs,
            self.server.global_model,
        )

        if self._window["enabled"] and self.cur_round == self._window["round"]:
            self.collect_validation_scores()

    def collect_validation_scores(self):
        validator_clients = sorted(self.list_clients)
        validator_loader = self.manager.create_batches(validator_clients)
        cosine_matrix = torch.zeros(
            (self.amount_of_clients, self.amount_of_clients), dtype=torch.float32
        )
        total_batches = len(validator_loader)
        total_sources = len(self.list_clients)
        start_time = time.time()

        print(
            f"Round {self.cur_round}: {len(validator_clients)} validator clients, "
            f"{total_sources} source models, {total_batches} validator batches",
            flush=True,
        )

        for batch_idx, clients_batch in enumerate(validator_loader, start=1):
            batch_ranks = [rank for _, rank in clients_batch]
            batch_start = time.time()
            self.manager.set_ranks_to_procs(clients_batch)
            print(
                f"  Validator batch {batch_idx}/{total_batches}: clients={batch_ranks}",
                flush=True,
            )

            for source_idx, source_rank in enumerate(self.list_clients, start=1):
                for pipe_num, _validator_rank in clients_batch:
                    self.server.send_content_to_client(
                        pipe_num,
                        {
                            "debug_validate_model": {
                                "source_rank": source_rank,
                                "model_state": self.server.client_models[source_rank],
                            }
                        },
                    )

                for pipe_num, _validator_rank in clients_batch:
                    response = self.server.rcv_content_from_client(pipe_num)
                    cosine_matrix[
                        response["validator_rank"], response["source_rank"]
                    ] = response["cosine"]

                if source_idx in {1, total_sources} or source_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    batch_elapsed = time.time() - batch_start
                    print(
                        f"    Source model {source_idx}/{total_sources} "
                        f"(rank={source_rank}) processed; "
                        f"batch_elapsed={batch_elapsed:.1f}s total_elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        self.server.trust_scores = self.server.print_validator_debug(
            cosine_matrix=cosine_matrix,
            validator_ranks=validator_clients,
            threshold=self._window["threshold"],
        )

    def aggregate(self):
        self.correction_map, self.make_corrections = (
            self.server.set_client_corrections()
        )
        if (
            self.num_steps_to_agg is not None
            and self.cur_round % self.num_steps_to_agg == 0
            and self.cur_round >= self.start_steps_to_agg
        ):
            print(f"Current Round {self.cur_round}. Make Client Aggregation.")
            self.server.aggregate_client_models()
        return self.server.global_model.state_dict()

    def get_communication_content(self, rank):
        content = {
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
            "strategy": self.strategy.get_client_payload(rank),
        }
        if getattr(self, "correction_map", None) is None:
            self.correction_map = {i: {} for i in range(self.amount_of_clients)}
            self.make_corrections = {i: False for i in range(self.amount_of_clients)}
        content["correction"] = (self.correction_map[rank], self.make_corrections[rank])
        if self.cur_round == 0:
            content["serialized"] = self.serialized_content
        else:
            missing_client_model = len(self.server.client_models[rank]) == 0
            refresh_cached_client_model = aggregated_client_models_in_prev_round(
                self.cur_round,
                self.num_steps_to_agg,
                self.start_steps_to_agg,
            )
            if missing_client_model:
                self.server.client_models[rank] = {
                    k: v.cpu() for k, v in self.server.global_model.state_dict().items()
                }
            if self.cfg.federated_params.cache_client_state.enabled:
                if refresh_cached_client_model:
                    content["aggregated_local_model"] = self.server.client_models[rank]
                elif missing_client_model:
                    content["client_model"] = self.server.client_models[rank]
            else:
                content["client_model"] = self.server.client_models[rank]
        content["cur_round"] = self.cur_round if self.global_decay else None
        return content
