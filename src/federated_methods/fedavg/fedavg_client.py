import sys
import copy
import time
import io
import torch
from collections import OrderedDict
from utils.data_utils import get_dataset_loader
from hydra.utils import instantiate

from utils.losses import get_loss
from utils.attack_utils import add_attack_functionality
from utils.process_utils import errors_child_handler
from utils.caching_utils import resolve_enabled_cache_objects
from utils.utils import tracked_state_items, get_tracked_model_state


class FedAvgClient:
    def __init__(self, *client_args, **client_kwargs):
        self._original_cls = type(self)
        self.client_args = client_args
        self.client_kwargs = client_kwargs
        cfg = self.client_args[0]
        dataset = self.client_args[1]
        self.cfg = cfg
        self.global_dataset = dataset
        self.rank = client_kwargs["rank"]
        self.pipe = client_kwargs["pipe"]
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.criterion = None
        self.server_model_state = None
        self.print_metrics = cfg.federated_params.print_client_metrics
        self.train_val_prop = cfg.federated_params.client_train_val_prop
        self.device = (
            "{}:{}".format(
                cfg.training_params.device,
                cfg.training_params.device_ids[
                    self.rank % len(cfg.training_params.device_ids)
                ],
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )

        if "model" in self.client_kwargs:
            assert "model_trainer" in self.client_kwargs
            self.model = self.client_kwargs.pop("model")
            self.model_trainer = self.client_kwargs.pop("model_trainer")
        else:
            self.model = instantiate(
                cfg.model, num_classes=self.global_dataset.num_classes
            )
            # Instantiate model_trainer which will be responsible for technical training of model
            self.model_trainer = instantiate(
                self.cfg.model_trainer, cfg=self.cfg, _recursive_=False
            )

        self.model.to(self.device)
        self._set_client_dataset()
        self._init_loaders()
        self._init_optimizer()
        self._init_criterion()
        self.pipe_commands_map = self.create_pipe_commands()
        self.cache_client_state = cfg.federated_params.cache_client_state.enabled
        if self.cache_client_state:
            self.clients_cache = client_kwargs.get("clients_cache", {})
            self.temporary_cache_rank = client_kwargs.get("temporary_cache_rank", None)
            self.cache_commands_map = self.create_cache_commands()
            self.enabled_cache_objects, self.cache_commands_map = (
                resolve_enabled_cache_objects(
                    self.cache_commands_map,
                    self.cfg.federated_params.cache_client_state,
                    warning_key="fedavg_client",
                )
            )
            self.pipe_commands_map["get_client_cache"] = self.get_client_cache
            self.pipe_commands_map["set_client_cache"] = self.set_client_cache
            self.restore_client_cache()

        self.grad = OrderedDict()
        self.local_epochs = self.cfg.federated_params.local_epochs

    def _init_optimizer(self):
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.train_dataset.data,
            num_classes=self.cfg.training_params.num_classes,
        )

    def _set_client_dataset(self):
        self.train_dataset = self.global_dataset.to_client_side(self.rank)

    def _init_loaders(self):
        self.valid_dataset = self.train_dataset.dataset_split(self.train_val_prop)

        self.train_loader = get_dataset_loader(
            self.train_dataset, self.cfg, drop_last=False
        )
        self.valid_loader = get_dataset_loader(
            self.valid_dataset, self.cfg, drop_last=False
        )

    def _set_attack_type(self, attack_content):
        self.attack_type = attack_content[0]
        self.attack_config = attack_content[1]

    def reinit_self(self, new_rank):
        # new_kwargs = dict(self.client_kwargs)
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

        # Recive content for local learning
        content = self.pipe.recv()
        self.parse_communication_content(content)

    def shutdown_self(self):
        print(f"Exit child {self.rank} process")
        sys.exit(0)

    def create_pipe_commands(self):
        # define a structure to process pipe values
        pipe_commands_map = {
            "serialized": self.deserialize,
            "update_model": lambda state_dict: self.model.load_state_dict(
                {k: v.to(self.device) for k, v in state_dict.items()}
            ),
            "attack_type": self._set_attack_type,
            "shutdown": lambda _: self.shutdown_self(),
            "reinit": lambda new_rank: self.reinit_self(new_rank),
        }

        return pipe_commands_map

    def create_cache_commands(self):
        return {
            "optimizer_state": lambda state_dict: self.optimizer.load_state_dict(
                state_dict
            ),
        }

    def create_client_cache(self):
        return {
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
        }

    def restore_client_cache(self):
        self.clients_cache = self.client_kwargs.get("clients_cache", {})
        if len(self.clients_cache) == 0:
            self.clients_cache = {
                rank: None for rank in range(self.cfg.federated_params.amount_of_clients)
            }
        client_state = self.clients_cache.get(self.rank)
        if client_state is None:
            return
        for key, value in client_state.items():
            if key in self.cache_commands_map:
                self.cache_commands_map[key](value)
            else:
                raise ValueError(
                    f"Recieved content in client {self.rank} from cache, with unknown key={key}"
                )

    def save_client_cache(self):
        self.clients_cache[self.rank] = {
            key: value
            for key, value in self.create_client_cache().items()
            if key in self.enabled_cache_objects
        }
        return self.clients_cache

    def get_client_cache(self, rank):
        response = copy.deepcopy(self.clients_cache[rank])
        self.pipe.send(response)
        content = self.pipe.recv()
        self.parse_communication_content(content)

    def set_client_cache(self, payload):
        self.clients_cache[payload["rank"]] = copy.deepcopy(payload["cache"])
        if self.temporary_cache_rank is not None:
            self.clients_cache[self.temporary_cache_rank] = None
        self.temporary_cache_rank = payload["rank"]
        content = self.pipe.recv()
        self.parse_communication_content(content)

    def deserialize(self, serialized_content):
        deserialized_content = {
            key: torch.load(io.BytesIO(payload), map_location="cpu")
            for key, payload in serialized_content.items()
        }
        self.parse_communication_content(deserialized_content)

    def get_loss_value(self, outputs, targets):
        return self.criterion(outputs, targets)

    def get_grad(self):
        self.model.eval()
        self.grad = OrderedDict()
        for key, tensor in tracked_state_items(self.model):
            self.grad[key] = tensor.detach().cpu() - self.server_model_state[key]

    def train(self):
        start = time.time()

        # Save the server model state to get_grad
        self.server_model_state = get_tracked_model_state(self.model)

        # Validate server weights before training to set up best model
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        # Train client
        self.model_trainer.train_fn(self)

        # Get client metrics
        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

        # Calculate client update
        self.get_grad()

        # Save training time
        self.result_time = time.time() - start

    def get_communication_content(self):
        # In fedavg_client we need to send only result of local learning
        result_dict = {
            "grad": self.grad,
            "rank": self.rank,
            "time": self.result_time,
            "server_metrics": (
                self.server_metrics,
                self.server_val_loss,
                len(self.valid_dataset),
            ),
        }
        if self.print_metrics:
            result_dict["client_metrics"] = (self.client_val_loss, self.client_metrics)

        return result_dict

    def parse_communication_content(self, content):
        # In fedavg_client we need to recive model after aggregate and
        # attack type for this client
        for key, value in content.items():
            if key in self.pipe_commands_map.keys():
                self.pipe_commands_map[key](copy.deepcopy(value))
            else:
                raise ValueError(
                    f"Recieved content in client {self.rank} from server, with unknown key={key}"
                )


@errors_child_handler
def multiprocess_client(*client_args, client_cls, pipe, rank, attack_type):
    # Init client instance
    client_kwargs = {"pipe": pipe, "rank": rank}
    client = client_cls(*client_args, **client_kwargs)

    # Loop of federated learning
    while True:
        # Wait content from server to start local learning
        content = client.pipe.recv()
        client.parse_communication_content(content)

        # Can be this realization of attack
        if client.attack_type != "no_attack":
            client = add_attack_functionality(
                client, client.attack_type, client.attack_config
            )

        client.train()

        # Send content to server, local learning ended
        content = client.get_communication_content()
        client.pipe.send(content)
