import copy
import random
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..personalized.client import PerClient
from .utils import resolve_global_decay
from utils.attack_utils import add_attack_functionality
from utils.process_utils import errors_child_handler
from utils.utils import get_tracked_model_state, tracked_named_parameters


class RoPOClient(PerClient):
    def __init__(self, *client_args, **client_kwargs):
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
        assert self.num_local_iters > 0
        self.correction = None
        self.make_correction = False
        self.debug_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training_params.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        self._debug_reference_state = None
        self._debug_reference_probs = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["correction"] = self.set_correction
        pipe_commands_map["client_model"] = self.set_client_model
        pipe_commands_map["aggregated_local_model"] = self.set_aggregated_local_model
        pipe_commands_map["cur_round"] = self.set_round
        pipe_commands_map["debug_validate_model"] = self.debug_validate_model
        return pipe_commands_map

    def create_cache_commands(self):
        cache_commands_map = super().create_cache_commands()
        cache_commands_map["model_state"] = self.set_client_model
        return cache_commands_map

    def create_client_cache(self):
        client_cache = super().create_client_cache()
        client_cache["model_state"] = {
            k: v.detach().cpu() for k, v in self.model.state_dict().items()
        }
        return client_cache

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

    def set_aggregated_local_model(self, client_model):
        self.set_client_model(client_model)
        if self.cache_client_state:
            self.clients_cache[self.rank] = {
                key: value
                for key, value in self.create_client_cache().items()
                if key in self.enabled_cache_objects
            }

    def set_round(self, cur_round):
        self.global_decay = resolve_global_decay(
            cur_round, self.use_global_decay, self.global_decay_mode
        )

    def reinit_self(self, new_rank):
        args = self.client_args
        kwargs = self.client_kwargs
        kwargs["rank"] = new_rank
        kwargs["model"] = self.model
        kwargs["model_trainer"] = self.model_trainer
        if self.cache_client_state:
            kwargs["clients_cache"] = self.save_client_cache()
            kwargs["temporary_cache_rank"] = self.temporary_cache_rank
        original_cls = self._original_cls
        self.__dict__.clear()
        original_cls.__init__(self, *args, **kwargs)

    def get_client_cache(self, rank):
        response = copy.deepcopy(self.clients_cache[rank])
        self.pipe.send(response)

    def set_client_cache(self, payload):
        self.clients_cache[payload["rank"]] = copy.deepcopy(payload["cache"])
        if self.temporary_cache_rank is not None:
            self.clients_cache[self.temporary_cache_rank] = None
        self.temporary_cache_rank = payload["rank"]

    def _evaluate_model_probs_on_debug_loader(self, model_state=None):
        if model_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in model_state.items()}
            )
        self.model.eval()
        probs_list = []
        with torch.no_grad():
            for _, (inputs, _targets) in self.debug_loader:
                inp = inputs[0].to(self.device)
                outputs = self.model(inp)
                probs_list.append(F.softmax(outputs, dim=-1).detach().cpu())
        return torch.cat(probs_list, dim=0)

    def debug_validate_model(self, payload):
        source_rank = int(payload["source_rank"])
        model_state = payload["model_state"]
        if self._debug_reference_state is None:
            self._debug_reference_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
        if self._debug_reference_probs is None:
            self._debug_reference_probs = self._evaluate_model_probs_on_debug_loader()

        candidate_probs = self._evaluate_model_probs_on_debug_loader(model_state)
        cosine = (
            F.cosine_similarity(candidate_probs, self._debug_reference_probs, dim=1)
            .mean()
            .item()
        )
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in self._debug_reference_state.items()}
        )
        return {
            "validator_rank": self.rank,
            "source_rank": source_rank,
            "cosine": cosine,
        }

    def _decayed_theta(self, local_step: int) -> float:
        return self.theta * (self.theta_decay**local_step) * self.global_decay

    def _apply_grad_correction(self, step_theta: float):
        if not self.make_correction or step_theta == 0 or self.correction is None:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            correction_tensor = self.correction.get(name)
            if correction_tensor is None:
                raise AssertionError(
                    f"Client {self.rank} param {name} has grad but no correction."
                )
            if correction_tensor.device != param.grad.device:
                correction_tensor = correction_tensor.to(param.grad.device)
            with torch.no_grad():
                param.grad.mul_(1 - step_theta)
                param.grad.add_(correction_tensor, alpha=step_theta)

    def _apply_param_correction(self, step_theta: float):
        if not self.make_correction or step_theta == 0 or self.correction is None:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            correction_tensor = self.correction.get(name)
            if correction_tensor is None:
                raise AssertionError(
                    f"Client {self.rank} param {name} has grad but no correction."
                )
            if correction_tensor.device != param.device:
                correction_tensor = correction_tensor.to(param.device)
            lr = self.optimizer.param_groups[0].get("lr", 1.0)
            with torch.no_grad():
                param.add_(correction_tensor, alpha=-step_theta * lr)

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
        self.server_model_state = get_tracked_model_state(self.model)
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )
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
            for name, param in tracked_named_parameters(self.model):
                server_param = self.server_model_state[name].to(param.device)
                update = param.data - server_param
                self.grad[name] = update.detach().cpu()

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["client_model"] = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }
        return result_dict


@errors_child_handler
def ropo_multiprocess_client(*client_args, client_cls, pipe, rank, attack_type, **_kwargs):
    client_kwargs = {"pipe": pipe, "rank": rank}
    client = client_cls(*client_args, **client_kwargs)

    while True:
        content = client.pipe.recv()
        if "debug_validate_model" in content:
            response = client.debug_validate_model(content["debug_validate_model"])
            client.pipe.send(response)
            continue

        client.parse_communication_content(content)
        if (
            "reinit" in content
            or "get_client_cache" in content
            or "set_client_cache" in content
        ):
            continue

        if getattr(client, "attack_type", "no_attack") != "no_attack":
            client = add_attack_functionality(
                client, client.attack_type, client.attack_config
            )

        client.train()
        response = client.get_communication_content()
        client.pipe.send(response)
