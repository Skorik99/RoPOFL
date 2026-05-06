from utils.caching_utils import get_runtime_caching


def tracked_named_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def tracked_state_items(model):
    for name, param in tracked_named_parameters(model):
        yield name, param

    for name, buffer in model.named_buffers():
        yield name, buffer


def get_tracked_model_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in tracked_state_items(model)
    }

def attach_runtime_caching(trainer, clustered=False):
    caching_variables = get_runtime_caching(
        type(trainer).__name__, bool(clustered)
    )
    if caching_variables is None:
        return

    trainer._runtime_caching = caching_variables["window"].copy()
    for key, value in caching_variables["overrides"].items():
        setattr(trainer, key, value)


def read_runtime_caching(trainer):
    caching_variables = getattr(trainer, "_runtime_caching", None)
    if caching_variables is None:
        return {"enabled": False, "round": None, "threshold": None}
    return dict(caching_variables)


def create_model_info(
    model_state, valid_metrics, valid_loss, test_metrics, test_loss, cfg
):
    model_info = {
        "model": model_state,
        "metrics": {
            "valid_metrics": valid_metrics,
            "valid_loss": valid_loss,
            "test_metrics": test_metrics,
            "test_loss": test_loss,
        },
        "config_file": cfg,
    }

    return model_info
