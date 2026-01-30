import os
import yaml


def get_target_dir(cfg, default_dir="cifar10"):
    target_dir = (
        cfg.test_dataset.get("download_path")
        or cfg.train_dataset.get("download_path")
        or default_dir
    )
    if not os.path.isabs(target_dir):
        target_dir = os.path.join(os.getcwd(), target_dir)
    print(f"Download path: {target_dir}")
    return target_dir


def save_map_files(train_df, test_df, target_dir):
    train_df.to_csv(os.path.join(target_dir, "train_map_file.csv"), index=False)
    test_df.to_csv(os.path.join(target_dir, "test_map_file.csv"), index=False)


def set_data_configs(target_path, config_names=["cifar10.yaml"]):
    print("Setting paths to .yaml files...\n")
    # HARDCODE paths
    config_dir = "src/configs/dataset/"
    if not os.path.isdir(config_dir):
        print(
            f"Directory {config_dir} not found. Set paths inside .yaml configs manually"
        )
        return

    if not os.path.isabs(target_path):
        curent_run_path = os.getcwd()
        target_path = os.path.join(curent_run_path, target_path)

    for filename in os.listdir(config_dir):
        if filename not in config_names:
            continue

        filepath = os.path.join(config_dir, filename)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data_sources = data.get("data_sources", {})

        if "test_map_file" in data_sources:
            test_map_path = [os.path.join(target_path, "test_map_file.csv")]
            data_sources["test_map_file"] = test_map_path

        if "train_map_file" in data_sources:
            train_map_path = [os.path.join(target_path, "train_map_file.csv")]
            data_sources["train_map_file"] = train_map_path

        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def set_base_path_configs(target_path, config_names=None):
    if config_names is None:
        config_names = []

    config_dir = "src/configs/dataset/"
    if not os.path.isdir(config_dir):
        print(
            f"Directory {config_dir} not found. Set paths inside .yaml configs manually"
        )
        return

    if not os.path.isabs(target_path):
        curent_run_path = os.getcwd()
        target_path = os.path.join(curent_run_path, target_path)

    for filename in os.listdir(config_dir):
        if config_names and filename not in config_names:
            continue

        filepath = os.path.join(config_dir, filename)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data["base_path"] = target_path

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def get_all_usernames():
    # Return all users in system
    with open("/etc/passwd", "r") as f:
        users = [line.split(":")[0] for line in f.readlines()]
    return users


def update_data_sources(base_path, data_sources):
    usernames = get_all_usernames()

    # We go through all data_sources and correct them in each path
    # (including path_to_zarr, filetered_map_files, etc.)
    for data_sources_type in data_sources.keys():
        for i in range(len(data_sources[data_sources_type])):
            if any(
                user in data_sources[data_sources_type][i].split("/")
                for user in usernames
            ):
                # if it leads to a user folder, then we don't update anything
                continue

            path_without_base = "/".join(
                data_sources[data_sources_type][i].split("/")[2:]
            )
            fpath = os.path.join("/", base_path, path_without_base)
            data_sources[data_sources_type][i] = fpath

    return data_sources
