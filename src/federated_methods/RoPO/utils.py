import math

import torch


def resolve_global_decay(cur_round, use_global_decay, global_decay_mode):
    if not use_global_decay:
        return 1.0
    if cur_round is None or cur_round <= 0:
        return 1.0
    if global_decay_mode == "log":
        decay = math.log10(cur_round + 1)
        decay = decay if decay > 1.0 else 1.0
        return 1.0 / decay
    if global_decay_mode is False:
        return 1.0
    raise ValueError(f"Unsupported global_decay mode: {global_decay_mode}")


def is_head_param(name):
    head_keys = ("head", "linear", "fc", "classifier")
    return any(
        name == key or name.startswith(f"{key}.") or f".{key}." in name
        for key in head_keys
    )


def should_exclude_head_updates(cfg, freeze_vit_head):
    target = getattr(cfg.model, "_target_", "") if cfg is not None else ""
    is_lora = "LoraVIT" in str(target)
    if not is_lora:
        return False
    if freeze_vit_head is None:
        head_freeze = getattr(cfg.model, "head_freeze", True)
        return not head_freeze
    return not freeze_vit_head


def build_trial_scores_matrix(trial_scores_map, amount_of_clients):
    if amount_of_clients == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    matrix = torch.zeros((amount_of_clients, amount_of_clients), dtype=torch.float32)
    for row in range(amount_of_clients):
        scores = trial_scores_map.get(row, [])
        offset = 0
        for col in range(amount_of_clients):
            if row == col:
                continue
            value = scores[offset] if offset < len(scores) else 0.0
            matrix[row, col] = value
            offset += 1
    return matrix


def pretty_print_trial_scores(amount_of_clients, trial_scores_map, matrix=None, title=None):
    if matrix is None:
        matrix = build_trial_scores_matrix(trial_scores_map, amount_of_clients)
        title = title or "Client Trial Scores"
    else:
        title = title or "Client Matrix"

    tensor = matrix.detach().cpu()
    print(f"{title}:")
    if tensor.numel() == 0:
        print("  <empty>")
        return

    num_clients = tensor.size(0)
    labels = [f"Client {i}" for i in range(num_clients)]
    header = f"{'Client':>8} | " + " ".join(f"{label:>8}" for label in labels)
    print(header)
    for row_idx, label in enumerate(labels):
        values = [f"{tensor[row_idx, col_idx].item():.3f}" for col_idx in range(num_clients)]
        print(f"{label:>8} | " + " ".join(f"{value:>8}" for value in values))


def aggregated_client_models_in_prev_round(
    cur_round, num_steps_to_agg, start_steps_to_agg
):
    if num_steps_to_agg is None or cur_round is None or cur_round <= 0:
        return False

    prev_round = cur_round - 1
    return prev_round % num_steps_to_agg == 0 and prev_round >= start_steps_to_agg
