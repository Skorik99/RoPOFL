import os
import datasets
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

from utils.dataset_utils import set_data_configs, get_target_dir, save_map_files
from .federated_dataset import FederatedDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Food101Dataset(FederatedDataset):
    def __init__(self, cfg, mode, data_sources, base_path, max_samples=None, **kwargs):
        # Setted transform for CIFAR-10 dataset
        self.transform = self.set_up_transform(mode)
        self.max_samples = max_samples

        super().__init__(cfg, mode, data_sources, base_path)

    def set_up_transform(self, mode):
        image_size = 224
        mean = (0.5493, 0.4451, 0.3435)
        std = (0.2731, 0.2759, 0.2799)
        if mode == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transform

    def downloading(self):
        print("We will download Food101 dataset!")
        print(
            "You can restart the experiment and set `+train_dataset.download_path` "
            "to specify the path to save the dataset."
        )

        target_dir = get_target_dir(self.cfg, default_dir="food101")

        # 1. Download dataset
        download_food101(target_dir)

        # 2. Convert to images, create map-files
        train_df, test_df = process_food101(target_dir)

        # 3. Update instantiated Dataset class
        self.target_dir = os.path.join(target_dir, "images")
        super().downloading()

        # 4. Save map-files
        save_map_files(train_df, test_df, self.target_dir)

        # 5. Update yaml configs
        set_data_configs(self.target_dir, config_names=["food101.yaml"])

    def preprocessing(self):
        # assert isinstance(self.max_samples, str), "max_samples should be str type"
        # max_samples = int(self.max_samples) if self.max_samples is not None else None
        max_samples = self.max_samples
        if not max_samples or max_samples <= 0:
            return

        if len(self.data) <= max_samples:
            return

        self.data = self.data.sample(
            n=max_samples, random_state=self.cfg.random_state
        ).reset_index(drop=True)
        print(f"Cropped Food101 {self.mode} dataset to {max_samples} samples.")

    def __getitem__(self, index):
        image = Image.open(self.data["fpath"][index])
        image = self.transform(image)
        label = self.data["target"][index]
        return index, ([image], label)


def download_food101(target_dir="food101"):
    os.makedirs(target_dir, exist_ok=True)
    print("Downloading Food101 dataset via HuggingFace datasets...")
    datasets.load_dataset("ethz/food101")


def save_food101_example(index, split, dataset, img_dir):
    try:
        item = dataset[index]
        label = item["label"]

        # only first 100 classes
        if label >= 100:
            return None

        image = item["image"].convert("RGB")
        filename = f"food101_{label}_{split}_{index:06d}.png"
        path = img_dir / filename
        image.save(path, format="png")

        return {
            "fpath": str(path),
            "file_name": filename,
            "target": label,
        }

    except Exception as e:
        print(f"[Error] index={index}, split={split}: {e}")
        return None


def process_food101_split(hf_dataset, split, img_dir, num_workers):
    records = []

    save_fn = partial(
        save_food101_example,
        split=split,
        dataset=hf_dataset,
        img_dir=img_dir,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_fn, i) for i in range(len(hf_dataset))]

        for future in tqdm(futures, desc=f"Saving {split}"):
            res = future.result()
            if res is not None:
                records.append(res)

    df = pd.DataFrame(records)
    df = df.sort_values(by="target").reset_index(drop=True)
    return df


def process_food101(base_dir="food101", num_workers=30):
    img_dir = Path(base_dir) / "images" / "data"
    img_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_absolute():
        img_dir = Path.cwd() / img_dir

    print("Processing Food101 with parallel saving (classes 0–99 only)...")

    dataset = datasets.load_dataset("ethz/food101")
    train_data = dataset["train"]
    test_data = dataset["validation"]

    train_df = process_food101_split(train_data, "train", img_dir, num_workers)
    test_df = process_food101_split(test_data, "test", img_dir, num_workers)

    return train_df, test_df
