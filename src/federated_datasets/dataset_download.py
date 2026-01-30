import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Allow running this file directly: python src/federated_datasets/dataset_download.py
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.dataset_utils import (
    get_target_dir,
    save_map_files,
    set_data_configs,
    set_base_path_configs,
)


def download_dataset(dataset_type, cfg):
    dataset_type = str(dataset_type).lower()

    if dataset_type == "cifar100":
        from federated_datasets.cifar100_dataset import (
            download_cifar100,
            process_cifar100,
        )

        print("We will download CIFAR-100 dataset!")
        target_dir = get_target_dir(cfg, default_dir="cifar100")

        download_cifar100(target_dir)
        train_df, test_df = process_cifar100(target_dir)

        target_dir = os.path.join(target_dir, "images")
        os.makedirs(target_dir, exist_ok=True)

        save_map_files(train_df, test_df, target_dir)
        set_data_configs(target_dir, config_names=["cifar100.yaml"])
        set_base_path_configs(target_dir, config_names=["cifar100.yaml"])
        return target_dir

    if dataset_type == "food101":
        from federated_datasets.food101_dataset import (
            download_food101,
            process_food101,
        )

        print("We will download Food101 dataset!")
        target_dir = get_target_dir(cfg, default_dir="food101")

        download_food101(target_dir)
        train_df, test_df = process_food101(target_dir)

        target_dir = os.path.join(target_dir, "images")
        os.makedirs(target_dir, exist_ok=True)

        save_map_files(train_df, test_df, target_dir)
        set_data_configs(target_dir, config_names=["food101.yaml"])
        set_base_path_configs(target_dir, config_names=["food101.yaml"])
        return target_dir

    raise ValueError(
        f"Unknown dataset_type={dataset_type!r}. Expected 'cifar100' or 'food101'."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets (map files + config updates)."
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["cifar100", "food101"],
        help="Dataset to download.",
    )
    parser.add_argument(
        "--download-path",
        default=None,
        help="Optional path to save the dataset (overrides default).",
    )
    return parser.parse_args()


def build_cfg(download_path):
    cfg = OmegaConf.create(
        {
            "train_dataset": {},
            "test_dataset": {},
        }
    )
    if download_path:
        cfg.train_dataset.download_path = download_path
        cfg.test_dataset.download_path = download_path
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = build_cfg(args.download_path)
    download_dataset(args.dataset_type, cfg)
