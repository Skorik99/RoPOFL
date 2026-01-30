import copy
import time

from ..fedavg.fedavg_client import FedAvgClient


class FineTuningClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.finetune_freeze_head = client_kwargs.get("finetune_freeze_head")
        self.finetune_freeze_backbone = client_kwargs.get("finetune_freeze_backbone")
        self.finetune_freeze_blocks = client_kwargs.get("finetune_freeze_blocks") or []
        self.finetune_best_metric = client_kwargs.get("finetune_best_metric", "loss")
        self.finetune_validate_per_epoch = client_kwargs.get(
            "finetune_validate_per_epoch", True
        )
        self.finetune_use_best_weights = client_kwargs.get(
            "finetune_use_best_weights", True
        )
        self.best_client_val_loss = None
        self.best_client_metrics = None
        self._default_requires_grad = {
            name: param.requires_grad for name, param in self.model.named_parameters()
        }
        self._resolve_finetune_policy()

    def _resolve_finetune_policy(self):
        dataset_target = str(getattr(self.cfg.train_dataset, "_target_", "")).lower()
        if "food101" in dataset_target:
            if self.finetune_freeze_backbone is None and not self.finetune_freeze_blocks:
                self.finetune_freeze_backbone = True
            if self.finetune_freeze_head is None:
                self.finetune_freeze_head = False
        if self.finetune_freeze_blocks is None:
            self.finetune_freeze_blocks = []
        self.finetune_freeze_blocks = [str(b) for b in self.finetune_freeze_blocks]

    def _is_head_param(self, name):
        head_keys = ("head", "linear", "fc", "classifier")
        return any(
            name == key or name.startswith(f"{key}.") or f".{key}." in name
            for key in head_keys
        )

    def _is_block_param(self, name, blocks):
        for block in blocks:
            if (
                name == block
                or name.startswith(f"{block}.")
                or f".{block}." in name
            ):
                return True
        return False

    def _apply_finetune_freeze(self):
        if (
            self.finetune_freeze_head is None
            and self.finetune_freeze_backbone is None
            and not self.finetune_freeze_blocks
        ):
            return

        for name, param in self.model.named_parameters():
            if self._is_head_param(name):
                param.requires_grad = not bool(self.finetune_freeze_head)
                continue

            if self.finetune_freeze_backbone:
                param.requires_grad = False
            elif self._is_block_param(name, self.finetune_freeze_blocks):
                param.requires_grad = False
            else:
                param.requires_grad = self._default_requires_grad.get(name, True)

        self._init_optimizer()

    def _normalized_metric_name(self):
        metric = str(self.finetune_best_metric).strip().lower()
        if metric in ("loss", "val_loss", "validation_loss"):
            return "loss"
        if metric in ("acc", "accuracy"):
            return "Accuracy"
        if metric in ("precision",):
            return "Precision"
        if metric in ("recall",):
            return "Recall"
        if metric in ("f1", "f1_score", "f1-score"):
            return "f1-score"
        return self.finetune_best_metric

    def _metric_value(self, metrics, metric_name):
        if metrics is None or metric_name not in metrics.index:
            return None
        return float(metrics.loc[metric_name].mean())

    def _is_better(self, val_loss, metrics, best_loss, best_metrics):
        metric_name = self._normalized_metric_name()
        if metric_name == "loss":
            return best_loss is None or val_loss < best_loss

        current = self._metric_value(metrics, metric_name)
        if current is None:
            return False

        best_val = self._metric_value(best_metrics, metric_name)
        if best_val is None:
            return True
        if current > best_val:
            return True
        if current == best_val and best_loss is not None:
            return val_loss < best_loss
        return False

    def _train_one_epoch(self):
        original_epochs = self.local_epochs
        self.local_epochs = 1
        self.model_trainer.train_fn(self)
        self.local_epochs = original_epochs

    def train(self):
        start = time.time()

        # Save the server model state to get_grad
        self.server_model_state = copy.deepcopy(self.model).state_dict()

        # Validate server weights before training to set up baseline metrics
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        self._apply_finetune_freeze()

        best_state = None
        best_loss = None
        best_metrics = None

        if self.finetune_validate_per_epoch:
            for _ in range(self.local_epochs):
                self._train_one_epoch()
                val_loss, metrics = self.model_trainer.client_eval_fn(self)
                if self._is_better(val_loss, metrics, best_loss, best_metrics):
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_loss = val_loss
                    best_metrics = metrics
        else:
            self.model_trainer.train_fn(self)
            best_loss, best_metrics = self.model_trainer.client_eval_fn(self)
            if self.finetune_use_best_weights:
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None and self.finetune_use_best_weights:
            self.model.load_state_dict(best_state)

        self.best_client_val_loss = best_loss
        self.best_client_metrics = best_metrics

        if self.print_metrics:
            self.client_val_loss = best_loss
            self.client_metrics = best_metrics

        # Calculate client update
        self.get_grad()

        # Save training time
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        if self.best_client_metrics is not None:
            result_dict["best_client_metrics"] = (
                self.best_client_val_loss,
                self.best_client_metrics,
            )
        return result_dict
