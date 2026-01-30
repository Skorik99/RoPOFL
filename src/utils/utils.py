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
