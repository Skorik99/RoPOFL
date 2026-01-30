from ..personalized.fedavg import PerFedAvg
from .ropo_client import RoPOClient
from .ropo_server import RoPOServer


class RoPO(PerFedAvg):
    def __init__(
        self,
        strategy,
        ckpt_path,
        server_test,
        beta,
        C,
        theta,
        num_local_iters,
        theta_decay,
        global_decay,
        sgd_correction,
        num_steps_to_agg,
        start_steps_to_agg,
        warmup_rounds=1,
        similarity_by_head=False,
        freeze_vit_head=None,
        print_trial_scores=False,
    ):
        super().__init__(strategy, ckpt_path, server_test)
        self.beta = beta
        self.theta = theta
        self.num_local_iters = num_local_iters
        self.theta_decay = theta_decay
        self.global_decay = global_decay
        self.sgd_correction = sgd_correction
        self.C = C
        self.num_steps_to_agg = num_steps_to_agg
        self.start_steps_to_agg = start_steps_to_agg
        self.warmup_rounds = warmup_rounds
        self.similarity_by_head = similarity_by_head
        self.freeze_vit_head = freeze_vit_head
        self.print_trial_scores = print_trial_scores
        if self.num_steps_to_agg is not None:
            assert (
                self.num_steps_to_agg > 0
            ), f"Number of communication rounds to aggregate client states should be more than 0, you provided: {self.num_steps_to_agg}"
            assert (
                self.start_steps_to_agg is not None
            ), f"When num_steps_to_agg is defined, you also need to define start_steps_to_agg attribute"
            assert (
                self.start_steps_to_agg >= 0
            ), f"Number of communication rounds to aggregate client states should be non-negative, you provided: {self.start_steps_to_agg}"

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = RoPOClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend(
            [
                self.theta,
                self.num_local_iters,
                self.theta_decay,
                self.global_decay,
                self.sgd_correction,
                self.global_decay,
            ]
        )

    def _init_server(self, cfg):
        # we do not test global model on server by default
        self.server_test = False
        byzt_clients = {
            idx
            for idx, attack_type in self.client_attack_map.items()
            if attack_type != "no_attack"
        }
        self.server = RoPOServer(
            cfg,
            self.server_test,
            self.beta,
            self.C,
            similarity_by_head=self.similarity_by_head,
            freeze_vit_head=self.freeze_vit_head,
            print_trial_scores=self.print_trial_scores,
            byzt_clients=byzt_clients,
        )

    def train_round(self):
        if (
            self.warmup_rounds > 0
            and self.cur_round is not None
            and self.cur_round < self.warmup_rounds
        ):
            self.list_clients = list(range(self.amount_of_clients))
            self.list_clients.sort()
            self.server.list_clients = self.list_clients
        return super().train_round()

    def aggregate(self):
        # We use aggregate method to set corrections to clients.
        self.correction_map, self.make_corrections = (
            self.server.set_client_corrections()
        )
        if (
            self.num_steps_to_agg is not None
            and self.cur_round % self.num_steps_to_agg == 0
            and self.cur_round >= self.start_steps_to_agg
        ):
            print(f"Current Round {self.cur_round}. Make Client Aggregation.")
            self.server.aggregate_client_models()
        # We do not aggregate model weights into a global model
        return self.server.global_model.state_dict()

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        # We send correction and local model to client
        if getattr(self, "correction_map", None) is None:
            # First round, no correction to send
            self.correction_map = {i: {} for i in range(self.amount_of_clients)}
            self.make_corrections = {i: False for i in range(self.amount_of_clients)}
        content["correction"] = (self.correction_map[rank], self.make_corrections[rank])
        if len(self.server.client_models[rank]) == 0:
            # First round, no client model to send to Client
            self.server.client_models[rank] = {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            }
        content["client_model"] = self.server.client_models[rank]
        if self.global_decay:
            content["cur_round"] = self.cur_round
        else:
            content["cur_round"] = None
        # We remove update model as we do not use global model in RoPO
        content.pop("update_model", None)
        return content
