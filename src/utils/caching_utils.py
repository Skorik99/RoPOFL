import io
import warnings

import torch


_EMITTED_CACHE_WARNINGS = set()


def serialize_payload(payload):
    cpu_payload = {k: v.detach().cpu() for k, v in payload.items()}
    buffer = io.BytesIO()
    torch.save(cpu_payload, buffer)
    return buffer.getvalue()


def resolve_enabled_cache_objects(
    cache_commands_map, cache_client_state, warning_key=None
):
    enabled_cache_objects = {
        key for key in cache_commands_map if cache_client_state.get(key, True)
    }
    unnamed_cache_objects = [
        key
        for key in cache_commands_map
        if key != "optimizer_state" and key not in cache_client_state
    ]

    cache_commands_map = {
        key: value
        for key, value in cache_commands_map.items()
        if key in enabled_cache_objects
    }

    if unnamed_cache_objects and warning_key not in _EMITTED_CACHE_WARNINGS:
        joined_objects = ", ".join(sorted(unnamed_cache_objects))
        warnings.warn(
            "cache_client_state.enabled=True enables unnamed cache objects by "
            f"default: {joined_objects}. Disable them explicitly if needed.",
            UserWarning,
            stacklevel=2,
        )
        _EMITTED_CACHE_WARNINGS.add(warning_key)

    return enabled_cache_objects, cache_commands_map


def build_rank_pipe_mapping(amount_of_clients, batch_size):
    rank_to_pipe = {}
    for rank in range(amount_of_clients):
        rank_to_pipe[rank] = rank % batch_size

    pipe_to_ranks = {pipe_idx: [] for pipe_idx in range(batch_size)}
    for rank, pipe_idx in rank_to_pipe.items():
        pipe_to_ranks[pipe_idx].append(rank)

    return rank_to_pipe, pipe_to_ranks


def assign_batch_to_pipes(clients_batch, rank_to_pipe, batch_size):
    free_pipes = set(range(batch_size))
    assignments = []
    overflow_ranks = []

    for rank in clients_batch:
        home_pipe = rank_to_pipe[rank]
        if home_pipe in free_pipes:
            assignments.append((home_pipe, rank))
            free_pipes.remove(home_pipe)
        else:
            overflow_ranks.append(rank)

    for rank in overflow_ranks:
        target_pipe = min(free_pipes)
        free_pipes.remove(target_pipe)
        assignments.append((target_pipe, rank))

    assignments.sort(key=lambda x: x[0])
    return assignments


def get_runtime_caching(method_name, clustered):
    return _RUNTIME_CACHING.get((method_name, bool(clustered)))


_RUNTIME_CACHING = {
    ("RoPO", True): {
        "window": {
            "enabled": True,
            "round": 15,
            "threshold": 0.65,
        },
        "overrides": {
            "C": 0.035,
            "beta": 0.99,
            "theta": 0.5,
            "num_local_iters": 8,
            "theta_decay": 0.95,
            "num_steps_to_agg": 1,
            "start_steps_to_agg": 15,
        },
    }
}
