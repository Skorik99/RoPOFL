import copy
import random
import time
from collections import OrderedDict
import math

import torch

from ..personalized.client import PerClient


class RoPOClient(PerClient):
    def __init__(self, *client_args, **client_kwargs):
        # self.client_args = client_args
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.theta = client_args[2]
        self.num_local_iters = client_args[3]
        self.theta_decay = client_args[4] if len(client_args) > 4 else 1.0
        self.global_decay_mode = client_args[5] if len(client_args) > 5 else False
        self.sgd_correction = client_args[6] if len(client_args) > 6 else True
        self.use_global_decay = bool(self.global_decay_mode)
        self.global_decay = 1.0
        assert (
            self.num_local_iters > 0
        ), f"num_local_iters must be positive, got {self.num_local_iters}"
        self.correction = None
        self.make_correction = False
        self._debug_prefix = f"[Client {self.rank}]"

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["correction"] = self.set_correction
        pipe_commands_map["client_model"] = self.set_client_model
        pipe_commands_map["cur_round"] = self.set_round
        return pipe_commands_map

    def set_correction(self, correction):
        correction_state, self.make_correction = correction
        if correction_state is None:
            self.correction = None
            return
        self.correction = correction_state.__class__(
            (key, value.to(self.device)) for key, value in correction_state.items()
        )

    def set_client_model(self, client_model):
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in client_model.items()}
        )

    def set_round(self, cur_round):
        if not self.use_global_decay:
            self.global_decay = 1.0
            return
        if cur_round is None or cur_round <= 0:
            self.global_decay = 1.0
            return
        if self.global_decay_mode == "log":
            decay = math.log10(cur_round + 1)
            decay = decay if decay > 1.0 else 1.0
            self.global_decay = 1.0 / decay
        elif self.global_decay_mode is False:
            self.global_decay = 1.0
        else:
            raise ValueError(f"Unsupported global_decay mode: {self.global_decay_mode}")

    def _decayed_theta(self, local_step: int) -> float:
        return self.theta * (self.theta_decay**local_step) * self.global_decay

    def _apply_grad_correction(self, step_theta: float):
        """Blend correction into gradients before optimizer.step (uses optimizer preconditioner)."""
        if not self.make_correction or step_theta == 0 or self.correction is None:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(f"Client {self.rank} Param {name} grad is None, skip correction")
                continue

            correction_tensor = self.correction.get(name)
            if correction_tensor is None:
                assert (
                    False
                ), f"Client {self.rank} Param {name} correction is None, but original grad is not None"

            if correction_tensor.device != param.grad.device:
                correction_tensor = correction_tensor.to(param.grad.device)

            theta_param = step_theta

            with torch.no_grad():
                param.grad.mul_(1 - theta_param)
                param.grad.add_(correction_tensor, alpha=theta_param)

    def _apply_param_correction(self, step_theta: float):
        """Apply correction directly in parameter space after optimizer.step (bypasses Adam moments)."""
        if not self.make_correction or step_theta == 0 or self.correction is None:
            return

        for name, param in self.model.named_parameters():
            grad_tensor = param.grad
            if grad_tensor is None:
                # print(f"Client {self.rank} Param {name} grad is None, skip correction")
                continue

            correction_tensor = self.correction.get(name)
            if correction_tensor is None:
                assert (
                    False
                ), f"Client {self.rank} Param {name} correction is None, but original grad is not None"

            if correction_tensor.device != param.device:
                correction_tensor = correction_tensor.to(param.device)

            theta_param = step_theta

            lr = self.optimizer.param_groups[0].get("lr", 1.0)
            with torch.no_grad():
                param.add_(correction_tensor, alpha=-theta_param * lr)

    def _choose_positions(self, total_batches: int):
        total_candidates = total_batches * max(1, self.local_epochs)
        steps_requested = self.num_local_iters or total_candidates
        steps_to_take = min(steps_requested, total_candidates)
        chosen_positions = set(random.sample(range(total_candidates), k=steps_to_take))
        return chosen_positions, steps_to_take

    def train_iter_fn(self, inputs, targets, step_theta: float):
        inp = inputs[0].to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inp)
        loss = self.get_loss_value(outputs, targets)
        loss.backward()
        if not self.sgd_correction:
            self._apply_grad_correction(step_theta)
        self.optimizer.step()
        if self.sgd_correction:
            self._apply_param_correction(step_theta)
        return loss.item()

    def train(self):
        start = time.time()
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        # --- ITERATIVE TRAINING ---
        self.model.train()
        total_batches = len(self.train_loader)
        chosen_positions, steps_to_take = self._choose_positions(total_batches)
        step_idx = 0
        for epoch in range(self.local_epochs):
            for batch_idx, (_, (inputs, targets)) in enumerate(self.train_loader):
                pos = epoch * total_batches + batch_idx
                if pos not in chosen_positions:
                    continue
                step_theta = self._decayed_theta(step_idx)
                self.train_iter_fn(inputs, targets, step_theta)
                step_idx += 1
                if step_idx >= steps_to_take:
                    break
            if step_idx >= steps_to_take:
                break

        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )
        self.get_grad()
        self.result_time = time.time() - start

    def get_grad(self):
        self.model.eval()
        self.grad = OrderedDict()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                server_param = self.server_model_state[name].to(param.device)
                update = param.data - server_param
                self.grad[name] = update.detach().cpu()
                # if torch.norm(update).item() != 0:
                #     print(f"Client {self.rank}", name, torch.norm(update).item())

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["client_model"] = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }
        return result_dict
