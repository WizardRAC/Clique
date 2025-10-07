import time
import typing

import bittensor as bt
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from CliqueAI.worker import get_worker
from common.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Miner neuron that requests work from the GPU worker.
    The miner does not perform calculations - it delegates to the worker.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        try:
            self.worker = get_worker()
            bt.logging.info("Miner connected to GPU worker successfully.")
        except Exception as e:
            bt.logging.error(f"Failed to initialize GPU worker: {e}")
            bt.logging.warning("Miner will use fallback NetworkX approximation.")
            self.worker = None
        
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        number_of_nodes = synapse.number_of_nodes
        adjacency_list = synapse.adjacency_list
        
        if self.worker is not None:
            try:
                bt.logging.info(
                    f"Requesting clique computation from worker for graph with {number_of_nodes} nodes"
                )
                
                # Worker performs all calculations
                maximum_clique = self.worker.compute_clique(adjacency_list, number_of_nodes)
                
                bt.logging.info(
                    f"Worker returned clique: {maximum_clique} with size {len(maximum_clique)}"
                )
            except Exception as e:
                bt.logging.error(f"Worker computation failed: {e}")
                bt.logging.warning("Falling back to NetworkX approximation")
                # Fallback to NetworkX if worker fails
                import networkx as nx
                dict_of_lists = {i: adjacency_list[i] for i in range(number_of_nodes)}
                graph = nx.from_dict_of_lists(dict_of_lists)
                maximum_clique = list(nx.approximation.max_clique(graph))
        else:
            # Fallback: use NetworkX approximation
            import networkx as nx
            dict_of_lists = {i: adjacency_list[i] for i in range(number_of_nodes)}
            graph = nx.from_dict_of_lists(dict_of_lists)
            maximum_clique = list(nx.approximation.max_clique(graph))
            bt.logging.info(
                f"NetworkX fallback - Maximum clique found: {maximum_clique} with size {len(maximum_clique)}"
            )
        
        synapse.adjacency_list = [[]]  # Clear up the adjacency list to reduce response size.
        synapse.maximum_clique = maximum_clique
        return synapse

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
