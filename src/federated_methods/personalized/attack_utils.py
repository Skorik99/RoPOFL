from typing import Dict, Iterable


def map_fixed_attack_clients(
    list_byzantines: Iterable[int],
    num_of_clients: int,
    attack_type: str,
) -> Dict[int, str]:
    """Map a fixed set of client ids to the provided attack type, others to no_attack."""
    byz_set = set(list_byzantines or [])
    return {
        idx: (attack_type if idx in byz_set else "no_attack")
        for idx in range(num_of_clients)
    }
