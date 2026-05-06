import warnings
from hydra.utils import instantiate
import torch.multiprocessing as mp
from federated_methods.fedavg.fedavg_client import multiprocess_client
from manager.batch_generator.cached import CachedBatchGenerator


class Manager:
    def __init__(self, cfg, server, df, batch_generator, **kwargs) -> None:
        self.server = server
        self.cfg = cfg
        self.cache_client_state = self.cfg.federated_params.cache_client_state.enabled
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.batch_generator = instantiate(
            batch_generator,
            amount_of_clients=self.amount_of_clients,
            df=df,
        )
        if (
            self.cache_client_state
            and type(self.batch_generator).__name__ != "CachedBatchGenerator"
        ):
            warnings.warn(
                "cache_client_state.enabled=True requires pipe-aware assignment. "
                "Switching batch generator to CachedBatchGenerator.",
                UserWarning,
                stacklevel=2,
            )
            self.batch_generator = CachedBatchGenerator(
                batch_size=self.batch_generator.batch_size,
                amount_of_clients=self.amount_of_clients,
            )

    def set_ranks_to_procs(self, clients_batch):
        if self.cache_client_state:
            transferred_cache = self.get_client_cache(clients_batch)
            self.set_client_cache(transferred_cache)

        # Step the manager to update ranks for clients
        for client_idx, new_rank in clients_batch:
            content = {"reinit": new_rank}
            self.server.send_content_to_client(client_idx, content)

    def create_batches(self, list_clients):
        self.batch_generator.create_batches(list_clients)
        return self.batch_generator.batches

    def get_client_cache(self, clients_batch):
        transferred_cache = []
        for pipe_num, rank in clients_batch:
            home_pipe = self.batch_generator.rank_to_pipe[rank]
            if pipe_num != home_pipe:
                content = {"get_client_cache": rank}
                self.server.send_content_to_client(home_pipe, content)
                response = self.server.rcv_content_from_client(home_pipe)
                transferred_cache.append(
                    {"pipe_num": pipe_num, "rank": rank, "cache": response}
                )
        return transferred_cache

    def set_client_cache(self, transferred_cache):
        for payload in transferred_cache:
            content = {
                "set_client_cache": {
                    "rank": payload["rank"],
                    "cache": payload["cache"],
                }
            }
            self.server.send_content_to_client(payload["pipe_num"], content)

    def create_clients(self, client_args, client_kwargs, attack_map):
        self.processes = []
        worker_target = client_kwargs.get("worker_target", multiprocess_client)

        # Init pipe for every client
        self.pipes = [mp.Pipe() for _ in range(self.batch_generator.batch_size)]
        self.server.pipes = [pipe[0] for pipe in self.pipes]  # Init input (server) pipe

        for pipe_num in range(self.batch_generator.batch_size):
            # Every process starts by calling the same function with the given arguments
            if hasattr(self.batch_generator, "pipe_to_ranks"):
                initial_rank = self.batch_generator.pipe_to_ranks[pipe_num][0]
            else:
                initial_rank = pipe_num
            process_kwargs = dict(client_kwargs)
            process_kwargs["pipe"] = self.pipes[pipe_num][1]
            process_kwargs["rank"] = initial_rank
            process_kwargs["attack_type"] = attack_map[initial_rank]
            p = mp.Process(
                target=worker_target,
                args=client_args,
                kwargs=process_kwargs,
            )
            p.start()
            self.processes.append(p)

    def stop_train(self):
        # Close all client proccesses
        for client_idx in range(self.batch_generator.batch_size):
            content = {"shutdown": None}
            self.server.send_content_to_client(client_idx, content)

        for p in self.processes:
            p.join()

        exit(0)
