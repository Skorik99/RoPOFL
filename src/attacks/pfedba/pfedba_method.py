import copy
from types import MethodType

from omegaconf import OmegaConf

from attacks.base.base_method import BaseAttackMethod
from .pfedba_server import PFedBAServer
from .pfedba_utils import (
    assert_supported_pfedba_setup,
    get_pfedba_byzantine_clients,
    is_personalized_pfedba_regime,
)
from utils.utils import attach_runtime_caching


class PFedBAMethod(BaseAttackMethod):
    def __init__(self, cfg=None, attack_client_cls=None, **attack_kwargs):
        super().__init__(cfg=cfg)
        self.attack_client_cls = attack_client_cls
        self.attack_cfg = OmegaConf.create(attack_kwargs)

    def change_functionality(self, trainer):
        attack_cfg = OmegaConf.create(
            OmegaConf.to_container(self.attack_cfg, resolve=True)
        )
        assert_supported_pfedba_setup(trainer.cfg, attack_cfg)
        is_personalized_regime = is_personalized_pfedba_regime(trainer)

        original_serialize_communication_content = trainer.serialize_communication_content
        original_get_communication_content = trainer.get_communication_content
        original_log_round = trainer.log_round
        original_cleanup = trainer.cleanup

        def wrapped_serialize_communication_content(this):
            self.prepare_pfedba_round_payload(this)
            return original_serialize_communication_content()

        def wrapped_get_communication_content(this, rank):
            content = original_get_communication_content(rank)
            should_attach_pfedba = (
                this.client_map_round[rank] == "pfedba" or is_personalized_regime
            )
            if should_attach_pfedba:
                client_attack_cfg = self.build_pfedba_client_cfg(
                    this,
                    pfedba_payload=copy.deepcopy(this.server.pfedba_payload),
                    is_byzantine=(this.client_map_round[rank] == "pfedba"),
                )
                content["attack_type"] = ("pfedba", client_attack_cfg)
            return content

        def wrapped_log_round(this):
            original_log_round()
            metrics = this.server.aggregate_pfedba_metrics(
                participating_clients=this.list_clients
            )
            if metrics is None:
                return

            print("\nPFedBA Server Validation Results:")
            print(metrics["clean_metrics"])
            print(f"PFedBA Server Validation Loss: {metrics['clean_loss']}")
            print("\nPFedBA Server ASR Results:")
            print(metrics["asr_metrics"])
            print(f"PFedBA Server ASR Loss: {metrics['asr_loss']}")
            print(f"PFedBA Server ASR: {metrics['asr']}", flush=True)

            if this.cfg.federated_params.print_client_metrics:
                client_metrics = this.server.aggregate_pfedba_metrics(
                    participating_clients=this.list_clients,
                    post_train=True,
                )
                if client_metrics is None:
                    return
                print("\nPFedBA Client Validation Results:")
                print(client_metrics["clean_metrics"])
                print(f"PFedBA Client Validation Loss: {client_metrics['clean_loss']}")
                print("\nPFedBA Client ASR Results:")
                print(client_metrics["asr_metrics"])
                print(f"PFedBA Client ASR Loss: {client_metrics['asr_loss']}")
                print(f"PFedBA Client ASR: {client_metrics['asr']}", flush=True)

        def wrapped_cleanup(this):
            original_cleanup()
            if hasattr(this.server, "reset_pfedba_metrics") and not is_personalized_regime:
                this.server.reset_pfedba_metrics()

        attack_server = PFedBAServer(
            byzantine_clients=get_pfedba_byzantine_clients(trainer),
            attack_cfg=attack_cfg,
            attack_client_cls=self.attack_client_cls,
            train_dataset=trainer.train_dataset,
            client_attack_map=trainer.client_attack_map,
            cfg=trainer.cfg,
            is_personalized_regime=is_personalized_regime,
        )
        trainer.server = attack_server(trainer.server)
        if is_personalized_regime:
            attach_runtime_caching(trainer, clustered=True)
        trainer.attack_configs["pfedba"] = self.build_pfedba_client_cfg(
            trainer, pfedba_payload=None, is_byzantine=True
        )
        trainer.serialize_communication_content = MethodType(
            wrapped_serialize_communication_content, trainer
        )
        trainer.get_communication_content = MethodType(
            wrapped_get_communication_content, trainer
        )
        trainer.log_round = MethodType(wrapped_log_round, trainer)
        trainer.cleanup = MethodType(wrapped_cleanup, trainer)
        return trainer

    def prepare_pfedba_round_payload(self, trainer):
        if "pfedba" not in trainer.client_map_round.values():
            trainer.server.pfedba_payload = None
            return
        trainer.server.train_pfedba_trigger()

    def build_pfedba_client_cfg(self, trainer, pfedba_payload, is_byzantine):
        return {
            "_target_": trainer.server.pfedba_attack_client_cls,
            "data_malicious_percent": trainer.server.pfedba_cfg.data_malicious_percent,
            "target_label": trainer.server.pfedba_target_label,
            "pfedba_payload": pfedba_payload,
            "is_byzantine": is_byzantine,
        }
