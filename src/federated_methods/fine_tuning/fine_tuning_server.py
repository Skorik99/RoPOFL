import pandas as pd

from ..fedavg.fedavg_server import FedAvgServer


class FineTuningServer(FedAvgServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_client_metrics = [
            pd.DataFrame() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.best_client_losses = [0 for _ in range(cfg.federated_params.amount_of_clients)]

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        best_pack = client_result.get("best_client_metrics")
        if not best_pack:
            return
        best_loss, best_metrics = best_pack
        self.best_client_losses[client_result["rank"]] = best_loss
        self.best_client_metrics[client_result["rank"]] = best_metrics
