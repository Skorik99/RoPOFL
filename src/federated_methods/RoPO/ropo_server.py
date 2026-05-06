from collections import OrderedDict
import os

import pandas as pd
import torch

from ..personalized.server import PerServer
from .utils import (
    build_trial_scores_matrix,
    is_head_param,
    pretty_print_trial_scores,
    should_exclude_head_updates,
)


class RoPOServer(PerServer):
    def __init__(
        self,
        cfg,
        server_test,
        beta,
        C,
        similarity_by_head=False,
        freeze_vit_head=None,
        print_trial_scores=False,
        byzt_clients=None,
    ):
        super().__init__(cfg, server_test, byzt_clients=byzt_clients)
        self.client_models = [OrderedDict() for _ in range(self.amount_of_clients)]
        self.trial_scores_map = {rank: [] for rank in range(self.amount_of_clients)}
        self.C = C
        self.beta = beta
        self.similarity_by_head = similarity_by_head
        self.freeze_vit_head = freeze_vit_head
        self.print_trial_scores = print_trial_scores
        self._exclude_head_updates = should_exclude_head_updates(
            self.cfg, self.freeze_vit_head
        )
        self._similarity_matrix = None
        self._score_matrix = None
        self.trust_scores = None

    def save_best_model(self, cur_round):
        pass

    def set_client_result(self, client_result):
        if client_result["rank"] in self.byzt_clients:
            param_names = {name for name, _ in self.global_model.named_parameters()}
            client_result["grad"] = OrderedDict(
                (k, v) for k, v in client_result["grad"].items() if k in param_names
            )

        super().set_client_result(client_result)
        self.client_models[client_result["rank"]] = client_result["client_model"]

    def update_trial_map(self):
        if len(self.trial_scores_map[0]) == 0:
            print("Set initial trial scores.")
        for rank in range(self.amount_of_clients):
            if rank not in self.active_ranks:
                continue
            if len(self.trial_scores_map[rank]) == 0:
                self.trial_scores_map[rank] = self.cur_trial_map[rank]
            else:
                self.trial_scores_map[rank] = [
                    self.beta * old_score + (1 - self.beta) * new_score
                    for old_score, new_score in zip(
                        self.trial_scores_map[rank], self.cur_trial_map[rank]
                    )
                ]
        if self.print_trial_scores:
            self.pretty_print_trial_scores()

    def _empty_scores(self):
        return [0.0 for _ in range(self.amount_of_clients - 1)]

    def set_client_scores(self):
        num_clients = self.amount_of_clients
        self.active_ranks = (
            self.list_clients
            if self.list_clients is not None
            else list(range(num_clients))
        )
        self.cur_trial_map = {rank: [] for rank in range(num_clients)}
        self.make_corrections = {rank: False for rank in range(num_clients)}

        if len(self.active_ranks) < 2:
            for rank in self.active_ranks:
                self.cur_trial_map[rank] = self._empty_scores()
            self.update_trial_map()
            return

        reference_params = list(self.global_model.named_parameters())
        if self.similarity_by_head:
            reference_params = [
                (name, param)
                for name, param in reference_params
                if is_head_param(name)
            ]
            if not reference_params:
                raise ValueError(
                    "similarity_by_head is True, but no head parameters were found."
                )

        flat_grads = []
        for rank in self.active_ranks:
            grad = self.client_gradients[rank]
            pieces = []
            for name, param in reference_params:
                tensor = grad.get(name)
                if tensor is None:
                    tensor = torch.zeros_like(param, device=param.device)
                pieces.append(tensor.view(-1))
            flat_grads.append(torch.cat(pieces))
        flat_grads = torch.stack(flat_grads)

        norms_sq = (flat_grads * flat_grads).sum(dim=1)
        similarity_matrix = flat_grads @ flat_grads.T
        threshold_matrix = self.C * norms_sq.unsqueeze(0)
        score_matrix = similarity_matrix - threshold_matrix
        score_matrix.fill_diagonal_(0)
        score_matrix = torch.where(
            score_matrix > 0, score_matrix, torch.zeros_like(score_matrix)
        )
        full_similarity = torch.zeros(
            (num_clients, num_clients), dtype=similarity_matrix.dtype
        )
        full_score = torch.zeros((num_clients, num_clients), dtype=score_matrix.dtype)
        for i, rank_i in enumerate(self.active_ranks):
            for j, rank_j in enumerate(self.active_ranks):
                full_similarity[rank_i, rank_j] = similarity_matrix[i, j]
                full_score[rank_i, rank_j] = score_matrix[i, j]
        self._similarity_matrix = full_similarity.detach().cpu()
        self._score_matrix = full_score.detach().cpu()
        if self.print_trial_scores:
            self.pretty_print_trial_scores(
                matrix=self._similarity_matrix, title="Client Similarity Matrix"
            )
            self.pretty_print_trial_scores(
                matrix=self._score_matrix, title="Client Score Matrix"
            )
        else:
            self._update_matrix_mean(
                self._similarity_matrix, "ropo_similarity_matrix_mean.pt"
            )
            self._update_matrix_mean(self._score_matrix, "ropo_score_matrix_mean.pt")

        active_index = {rank: idx for idx, rank in enumerate(self.active_ranks)}
        for rank in self.active_ranks:
            idx = active_index[rank]
            active_scores = torch.cat(
                (score_matrix[idx, :idx], score_matrix[idx, idx + 1 :])
            )
            active_peers = [r for r in self.active_ranks if r != rank]
            sum_scores = active_scores.sum().item()

            if sum_scores == 0:
                self.make_corrections[rank] = False
                score_map = (
                    {peer: 1 / len(active_peers) for peer in active_peers}
                    if active_peers
                    else {}
                )
            else:
                normalized_scores = (active_scores / sum_scores).tolist()
                score_map = {
                    peer: normalized_scores[i] for i, peer in enumerate(active_peers)
                }
                self.make_corrections[rank] = True

            full_scores = []
            for other in range(num_clients):
                if other == rank:
                    continue
                full_scores.append(score_map.get(other, 0.0))
            self.cur_trial_map[rank] = full_scores
        self.update_trial_map()

    def _update_matrix_mean(self, matrix, filename):
        run_dir = getattr(self.cfg, "single_run_dir", None)
        if not run_dir:
            return
        path = os.path.join(run_dir, filename)
        os.makedirs(run_dir, exist_ok=True)
        current = matrix.detach().cpu()
        payload = {"mean": current, "count": 1}
        if os.path.exists(path):
            try:
                existing = torch.load(path, map_location="cpu")
                prev_mean = existing.get("mean")
                prev_count = int(existing.get("count", 0))
                if prev_mean is not None and prev_count > 0:
                    new_count = prev_count + 1
                    payload = {
                        "mean": (prev_mean * prev_count + current) / new_count,
                        "count": new_count,
                    }
            except Exception:
                payload = {"mean": current, "count": 1}
        torch.save(payload, path)

    def calculate_client_corrections(self):
        num_clients = self.amount_of_clients
        correction_map = {rank: None for rank in range(num_clients)}
        active_ranks = [
            rank for rank in range(num_clients) if self.make_corrections[rank]
        ]
        if not active_ranks:
            return correction_map

        sample_rank = active_ranks[0]
        sample_grad = next(iter(self.client_gradients[sample_rank].values()))
        dtype = sample_grad.dtype
        device = sample_grad.device
        weights = torch.zeros((num_clients, num_clients), dtype=dtype, device=device)

        for rank in active_ranks:
            historical_scores = torch.tensor(
                self.trial_scores_map[rank], dtype=dtype, device=device
            )
            current_scores = torch.tensor(
                self.cur_trial_map[rank], dtype=dtype, device=device
            )
            active_mask = (current_scores != 0).to(dtype=dtype)
            filtered_scores = historical_scores * active_mask
            other_indices = list(range(rank)) + list(range(rank + 1, num_clients))
            if other_indices:
                weights[rank, other_indices] = filtered_scores

        for rank in active_ranks:
            correction_map[rank] = OrderedDict()

        for key in self.client_gradients[sample_rank].keys():
            if self._exclude_head_updates and is_head_param(key):
                for rank in active_ranks:
                    correction_map[rank][key] = torch.zeros_like(
                        self.client_gradients[rank][key]
                    )
                continue
            stacked_grads = torch.stack(
                [self.client_gradients[rank][key] for rank in range(num_clients)], dim=0
            ).to(device=device)
            aggregated = torch.einsum("ij,j...->i...", weights, stacked_grads)
            for rank in active_ranks:
                correction_map[rank][key] = aggregated[rank]
        return correction_map

    def set_client_corrections(self):
        self.set_client_scores()
        correction_map = self.calculate_client_corrections()
        return correction_map, self.make_corrections

    def aggregate_client_models(self):
        num_clients = self.amount_of_clients
        if num_clients == 0:
            return

        sample_param = None
        reference_model = None
        for model in self.client_models:
            if model:
                sample_param = next(iter(model.values()))
                reference_model = model
                break
        if sample_param is None or reference_model is None:
            return

        device = sample_param.device
        dtype = sample_param.dtype
        weight_matrix = torch.zeros(
            (num_clients, num_clients), dtype=dtype, device=device
        )

        if self.trust_scores is not None:
            for rank in range(num_clients):
                row_weights = self.trust_scores.get(str(rank))
                if row_weights is None:
                    weights = self.trial_scores_map[rank]
                    if not weights:
                        continue
                    other_indices = [idx for idx in range(num_clients) if idx != rank]
                    if other_indices:
                        weight_tensor = torch.tensor(weights, dtype=dtype, device=device)
                        weight_matrix[rank, other_indices] = weight_tensor
                else:
                    ordered_weights = [
                        float(row_weights.get(other_rank, 0.0))
                        for other_rank in range(num_clients)
                    ]
                    weight_matrix[rank] = torch.tensor(
                        ordered_weights, dtype=dtype, device=device
                    )
        else:
            for rank in range(num_clients):
                weights = self.trial_scores_map[rank]
                if not weights:
                    continue
                other_indices = [idx for idx in range(num_clients) if idx != rank]
                if other_indices:
                    weight_tensor = torch.tensor(weights, dtype=dtype, device=device)
                    weight_matrix[rank, other_indices] = weight_tensor

        updated_models = [OrderedDict() for _ in range(num_clients)]
        reference_keys = list(reference_model.keys())
        if self._exclude_head_updates:
            head_keys = {name for name in reference_keys if is_head_param(name)}
            for rank in range(num_clients):
                for name in head_keys:
                    if name in self.client_models[rank]:
                        updated_models[rank][name] = self.client_models[rank][name]
            reference_keys = [name for name in reference_keys if name not in head_keys]

        for name in reference_keys:
            tensors = []
            for client_model in self.client_models:
                tensor = client_model.get(name)
                if tensor is None:
                    tensor = torch.zeros_like(reference_model[name], device=device)
                else:
                    tensor = tensor.to(device=device)
                tensors.append(tensor)
            stacked_params = torch.stack(tensors, dim=0)
            if stacked_params.dtype != dtype:
                stacked_params = stacked_params.to(dtype=dtype)
            aggregated_params = torch.einsum("ij,j...->i...", weight_matrix, stacked_params)
            for rank in range(num_clients):
                updated_models[rank][name] = aggregated_params[rank]

        self.client_models = updated_models

    def cleanup(self):
        self.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]

    def pretty_print_trial_scores(self, matrix=None, title=None):
        if matrix is None:
            matrix = build_trial_scores_matrix(
                self.trial_scores_map, self.amount_of_clients
            )
            title = title or "Client Trial Scores"
        pretty_print_trial_scores(
            self.amount_of_clients, self.trial_scores_map, matrix=matrix, title=title
        )

    def _print_score_matrix(self, title, matrix, ordered_columns, row_labels):
        if not ordered_columns:
            return
        print(title)
        chunk_size = 20
        for start in range(0, len(ordered_columns), chunk_size):
            chunk = ordered_columns[start : start + chunk_size]
            header = "row   | " + " ".join(f"{rank:>7}" for rank in chunk)
            print(header)
            for row_label in row_labels:
                values = " ".join(f"{matrix[row_label][rank]:>7.4f}" for rank in chunk)
                print(f"{row_label:>4} | {values}")

    def _build_trust_score_matrix(
        self, cosine_matrix, validator_ranks, source_ranks, threshold
    ):
        thresholded_matrix = {}
        trust_score_matrix = {}
        for validator_rank in validator_ranks:
            cosines = torch.tensor(
                [cosine_matrix[validator_rank, source_rank] for source_rank in source_ranks],
                dtype=torch.float32,
            )
            masked = torch.where(cosines >= threshold, cosines, torch.zeros_like(cosines))
            thresholded_matrix[str(validator_rank)] = {
                source_rank: masked[idx].item()
                for idx, source_rank in enumerate(source_ranks)
            }
            denom = masked.sum()
            probs = masked / denom if denom.item() > 0 else torch.zeros_like(masked)
            trust_score_matrix[str(validator_rank)] = {
                source_rank: probs[idx].item()
                for idx, source_rank in enumerate(source_ranks)
            }
        return thresholded_matrix, trust_score_matrix

    def print_validator_debug(self, cosine_matrix, validator_ranks, threshold):
        validator_ranks = list(validator_ranks)
        source_ranks = list(range(self.amount_of_clients))
        row_labels = [str(rank) for rank in validator_ranks]
        cosine_matrix_dict = {
            str(validator_rank): {
                source_rank: cosine_matrix[validator_rank, source_rank].item()
                for source_rank in source_ranks
            }
            for validator_rank in validator_ranks
        }
        thresholded_matrix, trust_score_matrix = self._build_trust_score_matrix(
            cosine_matrix, validator_ranks, source_ranks, threshold
        )
        print(
            f"\nValidation scores for round {getattr(self, 'cur_round', None)}: "
            f"trust-score calculation with threshold={threshold}"
        )
        self._print_score_matrix(
            "  Cosine similarity matrix:", cosine_matrix_dict, source_ranks, row_labels
        )
        self._print_score_matrix(
            "  Thresholded cosine matrix:",
            thresholded_matrix,
            source_ranks,
            row_labels,
        )
        self._print_score_matrix(
            "  Trust-score matrix:", trust_score_matrix, source_ranks, row_labels
        )
        return trust_score_matrix
