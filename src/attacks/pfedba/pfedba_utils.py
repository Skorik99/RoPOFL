import copy
from typing import Iterable, Set

import numpy as np
import pandas as pd
from omegaconf import ListConfig
import torch
from torch.utils.data import DataLoader, Dataset


PATCH_SIZE = 8
IMAGE_SIZE = 32

_CIFAR_STATS = {
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2761),
    },
}


def get_dataset_name(cfg):
    target = str(cfg.train_dataset._target_).lower()
    if "cifar100" in target:
        return "cifar100"
    raise AssertionError("PFedBA currently supports only CIFAR-100 in the public repo.")


def assert_supported_pfedba_setup(cfg, attack_cfg):
    get_dataset_name(cfg)
    attack_scheme = cfg.federated_params.attack_scheme
    attack_types = cfg.federated_params.clients_attack_types
    if isinstance(attack_types, ListConfig):
        attack_types = list(attack_types)
    elif isinstance(attack_types, str):
        attack_types = [attack_types]
    assert attack_scheme == "constant", (
        f"PFedBA currently supports only constant attack scheme, got {attack_scheme}."
    )
    assert "pfedba" in attack_types, (
        "PFedBA requires cfg.federated_params.clients_attack_types to include `pfedba`."
    )
    assert 0.0 <= float(attack_cfg.data_malicious_percent) <= 1.0
    assert 0 <= int(attack_cfg.target_label) < cfg.training_params.num_classes
    assert int(attack_cfg.trigger_batch_size) > 0


def is_personalized_pfedba_regime(trainer):
    is_personalized = all(
        [
            hasattr(trainer, "define_clusters"),
            hasattr(trainer, "strategy"),
            hasattr(trainer, "server"),
            hasattr(trainer.server, "strategy_map"),
        ]
    )
    if is_personalized:
        assert (
            trainer.strategy == "sharded"
            or type(trainer.strategy).__name__ == "ShardedStrategy"
        ), "PFedBA personalized mode currently supports only sharded strategy."
    return is_personalized


def get_normalized_bounds(cfg, device):
    stats = _CIFAR_STATS[get_dataset_name(cfg)]
    mean = torch.tensor(stats["mean"], device=device).view(3, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper


def get_patch_coords(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    offset = (image_size - patch_size) // 2
    return offset, offset + patch_size


def apply_patch_trigger(inputs, delta_patch):
    patched = inputs.clone()
    y0, y1 = get_patch_coords()
    patched[:, :, y0:y1, y0:y1] = delta_patch.unsqueeze(0)
    return patched


def clone_dataset_with_df(dataset, df, mode=None):
    subset = copy.deepcopy(dataset)
    subset.data = df.reset_index(drop=True).copy()
    if mode is not None:
        subset.mode = mode
    return subset


def get_key_column(df):
    if "file_name" in df.columns:
        return "file_name"
    if "fpath" in df.columns:
        return "fpath"
    raise KeyError("PFedBA requires `file_name` or `fpath` column.")


def deterministic_poison_keys(
    df, rank, random_state, data_malicious_percent, target_label
) -> Set[str]:
    if len(df) == 0:
        return set()

    key_col = get_key_column(df)
    eligible_df = df[df["target"] != target_label].copy()
    if len(eligible_df) == 0:
        return set()

    eligible_keys = np.array(sorted(eligible_df[key_col].tolist()))
    num_poisoned = min(
        int(data_malicious_percent * len(df)),
        len(eligible_keys),
    )
    if num_poisoned <= 0:
        return set()

    rng = np.random.RandomState(int(random_state) + int(rank))
    chosen = rng.choice(eligible_keys, size=num_poisoned, replace=False)
    return set(chosen.tolist())


def poison_subset_from_keys(df, keys: Iterable[str]):
    key_set = set(keys)
    if len(key_set) == 0:
        return df.iloc[0:0].copy()
    key_col = get_key_column(df)
    return df[df[key_col].isin(key_set)].reset_index(drop=True).copy()


def filter_non_target(df, target_label):
    return df[df["target"] != target_label].reset_index(drop=True).copy()


def get_pfedba_byzantine_clients(method_instance):
    return sorted(
        rank
        for rank, attack_type in method_instance.client_attack_map.items()
        if attack_type == "pfedba"
    )


def build_pfedba_trigger_loader(server_instance):
    malicious_parts = []
    full_df = server_instance.pfedba_train_dataset.orig_data
    split_fn = server_instance.pfedba_train_dataset.train_val_split
    for rank in sorted(server_instance.pfedba_byzantine_clients):
        client_df = full_df[full_df["client"] == rank].reset_index(drop=True).copy()
        train_df, _ = split_fn(
            client_df,
            server_instance.cfg.federated_params.client_train_val_prop,
            server_instance.cfg.random_state,
        )
        poisoned_keys = deterministic_poison_keys(
            train_df,
            rank=rank,
            random_state=server_instance.cfg.random_state,
            data_malicious_percent=server_instance.pfedba_cfg.data_malicious_percent,
            target_label=server_instance.pfedba_target_label,
        )
        malicious_parts.append(poison_subset_from_keys(train_df, poisoned_keys))

    malicious_parts = [part for part in malicious_parts if len(part) > 0]
    if len(malicious_parts) == 0:
        return None

    trigger_dataset = clone_dataset_with_df(
        server_instance.pfedba_train_dataset,
        pd.concat(malicious_parts, ignore_index=True),
        mode="train",
    )
    return DataLoader(
        trigger_dataset,
        batch_size=server_instance.pfedba_cfg.trigger_batch_size,
        shuffle=True,
        num_workers=server_instance.cfg.training_params.num_workers,
        drop_last=False,
    )


def init_pfedba_train_loader(client_instance, data_malicious_percent, target_label):
    delta = client_instance.pfedba_payload["delta"].detach().cpu()
    poisoned_keys = deterministic_poison_keys(
        client_instance.train_dataset.data,
        rank=client_instance.rank,
        random_state=client_instance.cfg.random_state,
        data_malicious_percent=data_malicious_percent,
        target_label=target_label,
    )
    poisoned_dataset = PFedBAPoisonedDataset(
        client_instance.train_dataset,
        delta_patch=delta,
        target_label=target_label,
        poisoned_keys=poisoned_keys,
    )
    return DataLoader(
        poisoned_dataset,
        batch_size=client_instance.cfg.training_params.batch_size,
        shuffle=True,
        num_workers=client_instance.cfg.training_params.num_workers,
        drop_last=False,
    )


class PFedBAPoisonedDataset(Dataset):
    def __init__(self, base_dataset, delta_patch, target_label, poisoned_keys):
        self.base_dataset = base_dataset
        self.delta_patch = delta_patch.detach().cpu()
        self.target_label = int(target_label)
        self.poisoned_keys = set(poisoned_keys)
        self.data = base_dataset.data
        self.mode = base_dataset.mode
        self.num_classes = getattr(base_dataset, "num_classes", None)
        self._key_col = get_key_column(self.data)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample_index, (inputs, label) = self.base_dataset[index]
        image = inputs[0].clone()
        sample_key = self.data.iloc[index][self._key_col]
        if sample_key in self.poisoned_keys:
            image = apply_patch_trigger(image.unsqueeze(0), self.delta_patch).squeeze(0)
            label = self.target_label
        return sample_index, ([image], label)
