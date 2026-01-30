from collections import OrderedDict
import os
import pandas as pd

import torch

from ..personalized.server import PerServer


class FederatedClusteringServer(PerServer):
    def __init__(
        self,
        cfg,
        server_test,
        K,
        tau_percentile,
        clustering_iters,
        eta,
        debug,
        byzt_clients=None,
    ):
        super().__init__(cfg, server_test, byzt_clients=byzt_clients)
        self.client_models = [OrderedDict() for _ in range(self.amount_of_clients)]
        self.K = K
        self.tau_percentile = tau_percentile
        self.clustering_iters = clustering_iters
        self.eta = eta
        self.debug = debug
        self.cluster_centers = None
        self.cluster_assignments = None

    def save_best_model(self, cur_round):
        # We do not save best model for Federated-Clustering as we do not have a global model
        pass

    def set_client_result(self, client_result):
        # Strip buffer entries from byzantine gradients to keep shapes consistent
        if client_result["rank"] in self.byzt_clients:
            param_names = {name for name, _ in self.global_model.named_parameters()}
            client_result["grad"] = OrderedDict(
                (k, v) for k, v in client_result["grad"].items() if k in param_names
            )

        super().set_client_result(client_result)
        self.client_models[client_result["rank"]] = client_result["client_model"]

    def cluster_and_update_models(self):
        reference_params = list(self.global_model.named_parameters())
        flat_momentums = self._flatten_client_momentums(reference_params)
        self.cluster_centers = self._threshold_clustering(flat_momentums)
        self.cluster_assignments = self._assign_clusters(
            flat_momentums, self.cluster_centers
        )

        self._update_client_models(reference_params, self.cluster_centers)

    def _flatten_client_momentums(self, reference_params):
        flat = []
        for grad in self.client_gradients:
            pieces = []
            for name, _ in reference_params:
                tensor = grad.get(name)
                pieces.append(tensor.view(-1))
            flat.append(torch.cat(pieces))
        return torch.stack(flat)

    def _threshold_clustering(self, points):
        num_clients = points.size(0)
        if self.K > num_clients:
            raise ValueError(
                f"K={self.K} cannot exceed number of clients={num_clients}."
            )

        if self.cluster_centers is None:
            perm = torch.randperm(num_clients, device=points.device)
            centers = points[perm[: self.K]].clone()
        else:
            centers = self.cluster_centers.to(device=points.device).clone()

        for _ in range(self.clustering_iters):
            for k in range(self.K):
                center = centers[k]
                distances = torch.norm(points - center, dim=1)
                # if self.debug:
                #     key_distances = {
                #         idx: value for idx, value in enumerate(distances.tolist())
                #     }
                #     sorted_distances = sorted(
                #         key_distances.items(), key=lambda item: item[1]
                #     )
                #     print(
                #         "Cluster",
                #         k,
                #         "center update distances:",
                #         sorted_distances,
                #     )
                # tau is interpreted as a percentile over distances (e.g., 0.2 = 20th percentile).
                tau_value = self._percentile(distances, self.tau_percentile)
                in_ball = distances <= tau_value
                if in_ball.any():
                    sum_in = points[in_ball].sum(dim=0)
                    num_in = in_ball.sum().item()
                    centers[k] = (
                        sum_in + (num_clients - num_in) * center
                    ) / num_clients
        return centers

    def _assign_clusters(self, points, centers):
        distances = torch.stack(
            [torch.norm(points - center, dim=1) for center in centers]
        )
        assignments = torch.argmin(distances, dim=0).tolist()
        self._build_trial_score_matrix(distances, assignments)
        if self.debug:
            self.pretty_print_cluster_scores()
        else:
            self._update_matrix_mean(
                self._trial_score_matrix,
                "federated_clustering_trial_score_matrix_mean.pt",
            )
        return assignments

    def _build_trial_score_matrix(self, distances, assignments):
        # Build a client-by-cluster score matrix with softmax over assigned cluster only.
        num_clusters, num_clients = distances.shape
        scores = torch.zeros(
            (num_clients, num_clusters), dtype=distances.dtype, device=distances.device
        )
        for client_idx in range(num_clients):
            cluster_idx = assignments[client_idx]
            masked = torch.full(
                (num_clusters,),
                float("inf"),
                dtype=distances.dtype,
                device=distances.device,
            )
            masked[cluster_idx] = distances[cluster_idx, client_idx]
            scores[client_idx] = torch.softmax(-masked, dim=0)
        self._trial_score_matrix = scores.detach().cpu()

    def pretty_print_cluster_scores(self, decimals=3, title=None):
        data = getattr(self, "_trial_score_matrix", None)
        if data is None:
            return
        title = title or "Client Cluster Scores"
        num_clients, num_clusters = data.size(0), data.size(1)
        print(f"{title}:")
        if num_clients == 0:
            print("  <empty>")
            return

        row_labels = [f"Client {i}" for i in range(num_clients)]
        col_labels = [f"Cluster {i}" for i in range(num_clusters)]
        row_label_width = max(len("Client"), max(len(label) for label in row_labels))
        col_width = max(len(label) for label in col_labels)

        rendered_rows = []
        for row_idx in range(num_clients):
            rendered = []
            for col_idx in range(num_clusters):
                value = data[row_idx, col_idx].item()
                if abs(value) < 1e-12:
                    display = "0"
                else:
                    display = f"{value:.{int(decimals)}f}"
                col_width = max(col_width, len(display))
                rendered.append(display)
            rendered_rows.append(rendered)

        header = f"{'Client':>{row_label_width}} | " + " ".join(
            f"{label:>{col_width}}" for label in col_labels
        )
        print(header)
        for label, values in zip(row_labels, rendered_rows):
            row = " ".join(f"{value:>{col_width}}" for value in values)
            print(f"{label:>{row_label_width}} | {row}")

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

    def cleanup(self):
        # Keep gradients for partial participation; only reset metrics.
        self.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]

    def _update_client_models(self, reference_params, centers):
        param_names = {name for name, _ in reference_params}
        centers_by_param = [
            self._vector_to_param_dict(center, reference_params) for center in centers
        ]

        for rank in range(self.amount_of_clients):
            cluster_idx = self.cluster_assignments[rank]
            center_params = centers_by_param[cluster_idx]
            base_model = self.client_models[rank] or self.global_model.state_dict()
            updated = OrderedDict()
            for name, tensor in base_model.items():
                if name in param_names:
                    update = center_params[name].to(
                        device=tensor.device, dtype=tensor.dtype
                    )
                    with torch.no_grad():
                        updated[name] = tensor - self.eta * update
                else:
                    updated[name] = tensor
            self.client_models[rank] = updated

    @staticmethod
    def _vector_to_param_dict(vector, reference_params):
        output = OrderedDict()
        offset = 0
        for name, param in reference_params:
            numel = param.numel()
            output[name] = vector[offset : offset + numel].view_as(param)
            offset += numel
        return output

    @staticmethod
    def _percentile(values, q):
        if not 0 <= q <= 1:
            raise ValueError(f"percentile must be in [0,1], got {q}")
        flat = values.flatten()
        if flat.numel() == 0:
            return torch.tensor(0.0, device=values.device, dtype=values.dtype)
        n = flat.numel()
        if n == 1:
            return flat[0]
        idx = int(round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return flat.kthvalue(idx + 1).values
