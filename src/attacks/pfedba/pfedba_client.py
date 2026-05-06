import copy
import time
from types import MethodType

from torch.utils.data import DataLoader

from attacks.base.base_client import AttackClient
from utils.utils import get_tracked_model_state
from .pfedba_utils import (
    PFedBAPoisonedDataset,
    filter_non_target,
    get_key_column,
    init_pfedba_train_loader,
)


class PFedBAClientAttack(AttackClient):
    def __init__(
        self,
        data_malicious_percent,
        target_label,
        pfedba_payload=None,
        is_byzantine=None,
        **kwargs,
    ):
        self.data_malicious_percent = data_malicious_percent
        self.target_label = int(target_label)
        self.pfedba_payload = pfedba_payload
        self.is_byzantine = is_byzantine

    def apply_attack(self, client_instance):
        client_instance.pfedba_payload = self.pfedba_payload
        client_instance.pfedba_is_byzantine = self.is_byzantine
        client_instance._pfedba_base_get_communication_content = (
            client_instance.get_communication_content
        )
        client_instance.evaluate_pfedba_metrics = MethodType(
            PFedBAClientAttack.evaluate_pfedba_metrics, client_instance
        )
        client_instance.train = MethodType(PFedBAClientAttack.train, client_instance)
        client_instance.get_communication_content = MethodType(
            PFedBAClientAttack.get_communication_content, client_instance
        )

        client_instance.pfedba_server_asr_loss = None
        client_instance.pfedba_server_asr_metrics = None
        client_instance.pfedba_server_asr_len = 0
        client_instance.pfedba_client_asr_loss = None
        client_instance.pfedba_client_asr_metrics = None
        client_instance.pfedba_client_asr_len = 0
        if self.is_byzantine is not False:
            client_instance.train_loader = init_pfedba_train_loader(
                client_instance,
                data_malicious_percent=self.data_malicious_percent,
                target_label=self.target_label,
            )
        return client_instance

    def train(self):
        start = time.time()
        self.server_model_state = get_tracked_model_state(self.model)
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )
        (
            self.pfedba_server_asr_loss,
            self.pfedba_server_asr_metrics,
            self.pfedba_server_asr_len,
        ) = self.evaluate_pfedba_metrics()

        self.model_trainer.train_fn(self)

        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )
            (
                self.pfedba_client_asr_loss,
                self.pfedba_client_asr_metrics,
                self.pfedba_client_asr_len,
            ) = self.evaluate_pfedba_metrics()

        self.get_grad()
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = self._pfedba_base_get_communication_content()
        result_dict["pfedba_eval"] = {
            "server_asr": (
                self.pfedba_server_asr_metrics,
                self.pfedba_server_asr_loss,
                self.pfedba_server_asr_len,
            ),
            "client_asr": (
                self.pfedba_client_asr_metrics,
                self.pfedba_client_asr_loss,
                self.pfedba_client_asr_len,
            )
            if self.print_metrics
            else None,
        }
        return result_dict

    def evaluate_pfedba_metrics(self):
        delta = self.pfedba_payload["delta"].detach().cpu()
        target_label = int(self.pfedba_payload["target_label"])
        valid_df = filter_non_target(self.valid_dataset.data, target_label)
        if len(valid_df) == 0:
            return None, None, 0

        eval_dataset = copy.deepcopy(self.valid_dataset)
        eval_dataset.data = valid_df
        key_col = get_key_column(valid_df)
        poisoned_keys = set(valid_df[key_col].tolist())
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

        original_valid_dataset = self.valid_dataset
        original_valid_loader = self.valid_loader
        try:
            self.valid_dataset = eval_dataset
            self.valid_loader = eval_loader
            eval_loss, eval_metrics = self.model_trainer.client_eval_fn(self)
        finally:
            self.valid_dataset = original_valid_dataset
            self.valid_loader = original_valid_loader

        return eval_loss, eval_metrics, len(eval_dataset)
