from ..personalized.fedavg import PerFedAvg
from ..ditto.ditto_client import DittoClient
from ..ditto.ditto_server import DittoServer
from utils.caching_utils import serialize_payload


class Ditto(PerFedAvg):
    def __init__(self, strategy, ckpt_path, server_test, proximity):
        super().__init__(strategy, ckpt_path, server_test)
        self.proximity = proximity

    def _init_server(self, cfg):
        self.server = DittoServer(cfg, self.server_test)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = DittoClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.proximity])

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        if self.cfg.federated_params.cache_client_state.enabled:
            return content

        if self.cur_round == 0:
            local_model = content["serialized"]["update_model"]
        else:
            local_state = self.server.local_models[rank]
            if local_state is None:
                local_model = content["serialized"]["update_model"]
            else:
                local_model = serialize_payload(local_state)
        content["serialized"] = {**content["serialized"], "local_model": local_model}
        return content
