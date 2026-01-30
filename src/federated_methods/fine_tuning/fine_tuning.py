import torch

from ..fedavg.fedavg import FedAvg
from .fine_tuning_client import FineTuningClient
from .fine_tuning_server import FineTuningServer


class FineTuning(FedAvg):
    def __init__(
        self,
        ckpt_path=None,
        server_test=False,
        freeze_head=None,
        freeze_backbone=None,
        freeze_backbone_blocks=None,
        best_metric="loss",
        validate_per_epoch=True,
        use_best_weights=True,
    ):
        self.ckpt_path = ckpt_path
        self.server_test = server_test
        self.freeze_head = freeze_head
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_blocks = freeze_backbone_blocks
        self.best_metric = best_metric
        self.validate_per_epoch = validate_per_epoch
        self.use_best_weights = use_best_weights
        self._ckpt_loaded = False
        super().__init__()

    def _init_server(self, cfg):
        self.server = FineTuningServer(cfg)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FineTuningClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_kwargs["finetune_freeze_head"] = self.freeze_head
        self.client_kwargs["finetune_freeze_backbone"] = self.freeze_backbone
        self.client_kwargs["finetune_freeze_blocks"] = self.freeze_backbone_blocks
        self.client_kwargs["finetune_best_metric"] = self.best_metric
        self.client_kwargs["finetune_validate_per_epoch"] = self.validate_per_epoch
        self.client_kwargs["finetune_use_best_weights"] = self.use_best_weights

    def load_checkpoint(self):
        if not self.ckpt_path:
            return
        weights = torch.load(
            self.ckpt_path, map_location=self.server.device, weights_only=False
        )["model"]
        self.server.global_model.load_state_dict(weights)

    def get_communication_content(self, rank):
        if self.ckpt_path and not self._ckpt_loaded:
            self.load_checkpoint()
            self._ckpt_loaded = True
        return super().get_communication_content(rank)
