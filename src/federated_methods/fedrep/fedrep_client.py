import copy
import time
import torch
import torch.nn as nn
from hydra.utils import instantiate
from ..personalized.client import PerClient


class FedRepClient(PerClient):
    def __init__(
        self,
        *client_args,
        **client_kwargs,
    ):
        super().__init__(*client_args, **client_kwargs)
        self.warmup = True
        self._default_requires_grad = {
            name: param.requires_grad for name, param in self.model.named_parameters()
        }

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["update_model"] = self.load_body_new_head
        pipe_commands_map["warmup"] = self.set_wp
        return pipe_commands_map

    def set_wp(self, warmup_flag):
        self.warmup = warmup_flag

    def load_body_new_head(self, server_state_dict):
        new_state_dict = copy.deepcopy(server_state_dict)
        self.model.load_state_dict(new_state_dict)
        if not self.warmup:
            head_name, head_module = self._get_head_module()
            if isinstance(head_module, nn.Linear):
                new_head = nn.Linear(
                    head_module.in_features, self.global_dataset.num_classes
                ).to(self.device)
                setattr(self.model, head_name, new_head)

    def freeze_model(self, freeze_mode="unfreeze"):
        for name, param in self.model.named_parameters():
            if self._is_head_param(name):
                param.requires_grad = freeze_mode != "head"
            elif freeze_mode == "body":
                param.requires_grad = False
            else:
                param.requires_grad = self._default_requires_grad.get(name, True)

        self._init_optimizer()

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        start = time.time()

        # ---------- FedREP ---------- #
        if self.warmup:
            # Evaluate server model
            self.server_val_loss, self.server_metrics = (
                self.model_trainer.client_eval_fn(self)
            )
            # Just training server model
            self.model_trainer.train_fn(self)
        else:
            # Training head
            self.freeze_model(freeze_mode="body")  # freeze body, unfreeze head
            self.model_trainer.train_fn(self)

            # Evaluate personalized model
            self.server_val_loss, self.server_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

            # Training feature extractor
            self.local_epochs = 1
            self.freeze_model(freeze_mode="head")  # freeze head, unfreeze body
            self.model_trainer.train_fn(self)

        # ---------- FedREP ---------- #

        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

        self.get_grad()
        # Save training time
        self.result_time = time.time() - start

    def _is_head_param(self, name):
        head_keys = ("head", "linear", "fc", "classifier")
        return any(
            name == key or name.startswith(f"{key}.") or f".{key}." in name
            for key in head_keys
        )

    def _get_head_module(self):
        for name in ("head", "linear", "fc", "classifier"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                return name, module
        return None, None
