import pandas as pd

from ..personalized.server import PerServer


class FedAMPServer(PerServer):
    def cleanup(self):
        # Keep gradients for partial participation; only reset metrics.
        self.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
