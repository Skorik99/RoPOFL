import copy
import random
import time
from collections import OrderedDict

import torch

from ..personalized.client import PerClient


class FederatedClusteringClient(PerClient):
    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.alpha = client_args[2]
        self.num_local_iters = client_args[3]
        assert (
            self.num_local_iters > 0
        ), f"num_local_iters must be positive, got {self.num_local_iters}"
        assert 0 < self.alpha <= 1.0, f"alpha must be in (0,1], got {self.alpha}"
        self.momentum = None
        self._debug_prefix = f"[Client {self.rank}]"

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["client_model"] = self.set_client_model
        return pipe_commands_map

    def set_client_model(self, client_model):
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in client_model.items()}
        )

    def _choose_positions(self, total_batches: int):
        total_candidates = total_batches * max(1, self.local_epochs)
        steps_requested = self.num_local_iters or total_candidates
        steps_to_take = min(steps_requested, total_candidates)
        chosen_positions = set(random.sample(range(total_candidates), k=steps_to_take))
        return chosen_positions, steps_to_take

    def _ensure_momentum(self):
        if self.momentum is not None:
            return
        self.momentum = OrderedDict(
            (name, torch.zeros_like(param, device=self.device))
            for name, param in self.model.named_parameters()
        )

    def _update_momentum(self):
        self._ensure_momentum()
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            with torch.no_grad():
                self.momentum[name].mul_(1 - self.alpha)
                self.momentum[name].add_(param.grad, alpha=self.alpha)

    def train_iter_fn(self, inputs, targets):
        inp = inputs[0].to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inp)
        loss = self.get_loss_value(outputs, targets)
        loss.backward()
        self._update_momentum()
        return loss.item()

    def train(self):
        start = time.time()
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        # --- MOMENTUM COMPUTATION ---
        self.model.train()
        total_batches = len(self.train_loader)
        chosen_positions, steps_to_take = self._choose_positions(total_batches)
        step_idx = 0
        for epoch in range(self.local_epochs):
            for batch_idx, (_, (inputs, targets)) in enumerate(self.train_loader):
                pos = epoch * total_batches + batch_idx
                if pos not in chosen_positions:
                    continue
                self.train_iter_fn(inputs, targets)
                step_idx += 1
                if step_idx >= steps_to_take:
                    break
            if step_idx >= steps_to_take:
                break

        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

        # Store momentum as the "grad" payload
        self.grad = OrderedDict(
            (k, v.detach().cpu()) for k, v in (self.momentum or {}).items()
        )
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["client_model"] = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }
        return result_dict
