from .base import Base
from utils.caching_utils import assign_batch_to_pipes, build_rank_pipe_mapping


class CachedBatchGenerator(Base):
    def __init__(self, batch_size, amount_of_clients, *args, **kwargs):
        super().__init__(batch_size, amount_of_clients)
        self.rank_to_pipe, self.pipe_to_ranks = build_rank_pipe_mapping(
            self.amount_of_clients, self.batch_size
        )

    def create_batches(self, current_round_clients):
        base_batches = [
            current_round_clients[i : i + self.batch_size]
            for i in range(0, len(current_round_clients), self.batch_size)
        ]

        self.batches = [
            assign_batch_to_pipes(batch, self.rank_to_pipe, self.batch_size)
            for batch in base_batches
        ]
        self.num_batches = len(self.batches)
