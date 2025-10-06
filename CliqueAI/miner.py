import time
import typing
import random
import threading
from itertools import combinations, islice
from typing import List
import heapq
import collections
from collections import defaultdict
import signal

import bittensor as bt
import networkx as nx
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron

try:
    import torch
    from CliqueAI.gpu_clique_finder import GPUCliqueFinder
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    bt.logging.warning("PyTorch not available, GPU acceleration disabled")


class Miner(BaseMinerNeuron):
    """
    Optimized miner with advanced clique-finding algorithms for better performance.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )
        self.random_seed = random.randint(0, 1000000)
        
        if GPU_AVAILABLE:
            self.gpu_finder = GPUCliqueFinder()
            bt.logging.info("GPU-accelerated clique finding enabled")
        else:
            self.gpu_finder = None
            bt.logging.info("Using CPU-only clique finding")
    
    def _get_thread_safe_seed(self):
        """Generate thread-safe random seed for concurrent execution"""
        thread_id = threading.get_ident()
        current_time = time.time()
        # Combine thread ID, time, and instance seed for uniqueness
        return int((current_time * 1000000 + thread_id + self.random_seed) % 1000000)

    def is_valid_clique(self, graph, nodes: List[int]) -> bool:
        """
        Fast clique validation without checking maximality.
        """
        if len(nodes) <= 1:
            return len(nodes) == 1
        
        node_set = set(nodes)
        if len(node_set) != len(nodes):
            return False
        
        # Check if all pairs are connected
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not graph.has_edge(nodes[i], nodes[j]):
                    return False
        return True
    
    def is_maximal_clique(self, graph, nodes: List[int]) -> bool:
        """
        Check if the given clique is maximal (no additional node can be added).
        """
        if not nodes:
            return False
        if not self.is_valid_clique(graph, nodes):
            return False
        node_set = set(nodes)
        for candidate in graph.nodes():
            if candidate in node_set:
                continue
            # candidate must connect to all nodes in current clique to extend it
            ok = True
            for u in nodes:
                if not graph.has_edge(candidate, u):
                    ok = False
                    break
            if ok:
                return False
        return True

    def make_clique_maximal(self, graph, nodes: List[int]) -> List[int]:
        """
        Expand a valid clique to a maximal clique by greedily adding common neighbors.
        Returns a sorted list.
        """
        if not nodes:
            return []
        # If input is not a clique, attempt to repair by shrinking to a clique
        clique = []
        for v in nodes:
            if all(graph.has_edge(v, u) for u in clique):
                clique.append(v)
        # Greedy expansion to maximal
        candidates = set(graph.nodes()) - set(clique)
        while True:
            addable = [v for v in candidates if all(graph.has_edge(v, u) for u in clique)]
            if not addable:
                break
            # choose the node with highest degree among addable to promote larger cliques
            v = max(addable, key=lambda x: graph.degree(x))
            clique.append(v)
            candidates.remove(v)
        return sorted(clique)
    
    def find_maximum_clique_degeneracy(self, graph):
        """
        Added degeneracy-based algorithm for large graphs
        """
        if graph.number_of_nodes() == 0:
            return []
        
        # Compute degeneracy ordering using proper algorithm
        nodes = list(graph.nodes())
        if len(nodes) == 1:
            return nodes
        
        # Create a copy of the graph to modify
        temp_graph = graph.copy()
        ordering = []
        
        # Build degeneracy ordering by repeatedly removing minimum degree vertex
        while temp_graph.nodes():
            # Find vertex with minimum degree
            min_vertex = min(temp_graph.nodes(), key=lambda v: temp_graph.degree(v))
            ordering.append(min_vertex)
            temp_graph.remove_node(min_vertex)
        
        # Find maximum clique using degeneracy ordering
        best_clique = []
        
        for i, v in enumerate(ordering):
            # Consider only later vertices in ordering that are neighbors of v
            later_vertices = set(ordering[i+1:])
            candidates = [u for u in graph.neighbors(v) if u in later_vertices]
            
            if not candidates:
                if len([v]) > len(best_clique):
                    best_clique = [v]
                continue
            
            # Find maximum clique starting with v
            current_clique = [v]
            remaining_candidates = candidates[:]
            
            # Greedy expansion
            while remaining_candidates:
                # Find candidate connected to all vertices in current clique
                next_vertex = None
                for candidate in remaining_candidates:
                    if all(graph.has_edge(candidate, u) for u in current_clique):
                        next_vertex = candidate
                        break
                
                if next_vertex is None:
                    break
                
                current_clique.append(next_vertex)
                remaining_candidates.remove(next_vertex)
                # Filter remaining candidates to only those connected to new vertex
                remaining_candidates = [u for u in remaining_candidates if graph.has_edge(next_vertex, u)]
            
            if len(current_clique) > len(best_clique):
                best_clique = current_clique
        
        return best_clique

    def find_maximum_clique_local_search(self, graph, max_iterations=1000):
        """
        Added local search algorithm with random restarts
        """
        if graph.number_of_nodes() == 0:
            return []
        
        random.seed(self.random_seed + hash(str(sorted(graph.edges()))))
        best_clique = []
        
        for restart in range(5):  # Multiple restarts
            # Start with random vertex
            current = [random.choice(list(graph.nodes()))]
            
            for iteration in range(max_iterations // 5):
                improved = False
                
                # Try to add vertices
                candidates = []
                for v in graph.nodes():
                    if v not in current and all(graph.has_edge(v, u) for u in current):
                        candidates.append(v)
                
                if candidates:
                    # Add vertex that maximizes connections to non-clique vertices
                    best_candidate = max(candidates, key=lambda v: graph.degree(v))
                    current.append(best_candidate)
                    improved = True
                
                if not improved:
                    break
            
            if len(current) > len(best_clique):
                best_clique = current[:]
        
        return best_clique
    
    def find_maximum_clique_sam_greedy_improved(self, graph, rng):
        """
        Drop-in upgrade: randomized greedy (RCL) + 1-swap + re-expand.
        Prints 'true' if the returned set is fully connected (a clique), else 'false'.
        Returns the clique (list of nodes).
        """

        if graph.number_of_nodes() == 0:
            print("true")  # empty clique is trivially a clique
            return []

        # --- parameters ---
        RCL_FRAC = 0.20          # pick from top 20% by residual degree
        MAX_1SWAP_TRIES = 5#200    # cap inner improvements to guarantee termination
        
        # try:
        #     rng.seed(self._get_thread_safe_seed())
        # except Exception:
        #     pass
        # adjacency as sets for fast checks
        adj = {u: set(graph.neighbors(u)) for u in graph}
        V = list(graph.nodes())
        deg = {u: len(adj[u]) for u in V}
        def greedy_reexpand(C, Cset):
            """Greedy grow using RCL on residual degree inside current candidate set."""
            if not C:
                cand = set(V)
            else:
                it = iter(C)
                cand = adj[next(it)].copy()
                for u in it:
                    cand &= adj[u]
            cand -= Cset

            while cand:
                scored = [(v, len(adj[v] & cand)) for v in cand]
                scored.sort(key=lambda t: t[1], reverse=True)
                k = max(1, int(len(scored) * RCL_FRAC))
                pick = rng.choice([v for v, _ in scored[:k]])
                C.append(pick); Cset.add(pick)
                cand &= adj[pick]
                cand.discard(pick)
            return C, Cset

        def one_swap_improve(C, Cset):
            """Try a single 1-swap (replace one vertex by an outsider), then re-expand."""
            outside = [v for v in V if v not in Cset]
            rng.shuffle(outside)
            for v in outside:
                missing = [u for u in C if v not in adj[u]]
                if len(missing) == 1:
                    u = missing[0]
                    C.remove(u); Cset.remove(u)
                    C.append(v); Cset.add(v)
                    greedy_reexpand(C, Cset)
                    return True
            return False

        best_clique = []
        #### method 1
        # starts = list(graph.nodes())[:min(50, graph.number_of_nodes())]
        # starts.sort(key=lambda u: deg[u], reverse=True)

        #### method 2
        # k = min(50, graph.number_of_nodes())
        # deg_view = dict(graph.degree())
        # nodes = list(deg_view)
        # weights = [max(1, deg_view[u])**1.2 for u in nodes]  # α = 1.2
        # starts = []
        # pool = nodes[:]
        # w = weights[:]
        # for _ in range(k):
        #     r, s = rng.random() * sum(w), 0.0
        #     for i, wt in enumerate(w):
        #         s += wt
        #         if s >= r:
        #             starts.append(pool[i])
        #             # remove chosen item to sample without replacement
        #             pool.pop(i); w.pop(i)
        #             break
        # #print(f"starts-{starts}")

        #### method3
        k = min(50, graph.number_of_nodes())
        nodes = list(graph.nodes())
        # use your RNG if you have one; otherwise fall back to random
        starts = rng.sample(nodes, k)          # without replacement
        starts.sort(key=lambda u: deg[u], reverse=True)
        #########################################
        for start_node in starts:
            C = [start_node]
            Cset = {start_node}
            greedy_reexpand(C, Cset)

            tries = 0
            while tries < MAX_1SWAP_TRIES:
                tries += 1
                if not one_swap_improve(C, Cset):
                    break

            if len(C) > len(best_clique):
                best_clique = C[:]

        # ---- full connectivity check (print true/false) ----
        is_clique = all(
            (v in adj[u])
            for i, u in enumerate(best_clique)
            for v in best_clique[i+1:]
        )
        # print("true" if is_clique else "false")

        return best_clique

    def find_maximum_clique_hybrid(self, graph):
        """
        Enhanced hybrid approach with multiple advanced algorithms
        """
        num_nodes = graph.number_of_nodes()
        
        if num_nodes == 0:
            return {"winner": [], "method": "empty_graph", "all_results": {}}
        
        if GPU_AVAILABLE and self.gpu_finder is not None:
            try:
                bt.logging.info("Using GPU-accelerated parallel search")
                adj_matrix = self.gpu_finder.graph_to_adjacency_matrix(graph)
                gpu_result = self.gpu_finder.parallel_greedy_search(
                    adj_matrix,
                    num_searches=200,
                    timeout=17.0,
                    seed=self._get_thread_safe_seed()
                )
                
                gpu_clique = gpu_result["clique"]
                
                # Verify and make maximal using NetworkX
                if gpu_clique and self.is_valid_clique(graph, gpu_clique):
                    gpu_clique = self.make_clique_maximal(graph, gpu_clique)
                    
                    if self.is_maximal_clique(graph, gpu_clique):
                        bt.logging.info(f"GPU found maximal clique of size {len(gpu_clique)}")
                        bt.logging.info(f"GPU stats: {gpu_result['num_searches']} searches, "
                                      f"{gpu_result['unique_cliques']} unique cliques")
                        
                        return {
                            "winner": gpu_clique,
                            "method": "gpu_parallel_greedy",
                            "all_results": {"gpu_parallel_greedy": {"clique": gpu_clique, "score": len(gpu_clique)}}
                        }
            except Exception as e:
                bt.logging.error(f"GPU acceleration failed: {e}, falling back to CPU")
        
        # Choose algorithms based on graph size
        results = {}
        
        # Always use these algorithms
        start_time = time.time()
        timeout = 17
        
        rng = random.Random(self._get_thread_safe_seed())
        
        # before the loop
        num = 0
        uniq_counts = {}                  # frozenset(clique) -> occurrences
        size_hist = collections.Counter() # size -> count
        sizes = []                     # per-run sizes
        
        while num <= 200:
            if time.time() - start_time >= timeout:
                bt.logging.warning(f"Timeout reached during greedy iteration {num}")
                break

            greedy_clique = self.find_maximum_clique_sam_greedy_improved(graph, rng)
            greedy_clique = self.make_clique_maximal(graph, greedy_clique)

            key = frozenset(greedy_clique)            # order-insensitive identity
            uniq_counts[key] = uniq_counts.get(key, 0) + 1
            k = len(greedy_clique)
            size_hist[k] += 1
            sizes.append(k)
            is_dup = (uniq_counts[key] > 1)

            num = num + 1
        
        if uniq_counts:
            # best: largest size, lowest frequency
            best_clique, best_occ = max(uniq_counts.items(), key=lambda kv: (len(kv[0]), -kv[1]))
            print(f'[best] clique={sorted(best_clique)} size={len(best_clique)} occ={best_occ}')

            # rare-large: lowest frequency, largest size
            rare_large, rl_occ = min(uniq_counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
            print(f'[rare-large] clique={sorted(rare_large)} size={len(rare_large)} occ={rl_occ}')
            
            if len(rare_large) >= len(best_clique):                
                results["RSLgreedy_improved"] = {"clique": sorted(rare_large), "score": len(rare_large)}
            elif rl_occ >= best_occ:
                results["RSLgreedy_improved"] = {"clique": sorted(best_clique), "score": len(best_clique)}
            else:
                results["RSLgreedy_improved"] = {"clique": sorted(rare_large), "score": len(rare_large)}
                
            any_dups = False
            for key, cnt in sorted(uniq_counts.items(), key=lambda kv: (-kv[1], -len(kv[0]))):
                if cnt > 1:
                    any_dups = True    
                print(f'  {sorted(key)}--{len(sorted(key))} -> {cnt}')
                
            if not any_dups:
                print('  (no duplicates)')
        else:
            print('[summary] no cliques found')
        
        degeneracy_clique = self.find_maximum_clique_degeneracy(graph)
        degeneracy_clique = self.make_clique_maximal(graph, degeneracy_clique)
        results["degeneracy"] = {"clique": degeneracy_clique, "score": len(degeneracy_clique)}
        
        # NetworkX approximation as fallback
        try:
            approx_clique = list(nx.approximation.max_clique(graph))
            approx_clique = self.make_clique_maximal(graph, approx_clique)
            results["approximation"] = {"clique": approx_clique, "score": len(approx_clique)}
        except:
            results["approximation"] = {"clique": [], "score": 0}
            
        # Validate maximality and clique property; compute final scores
        for method_name, result in list(results.items()):
            clique = result["clique"]
            if not self.is_maximal_clique(graph, clique):
                # try to maximalize once more as safeguard
                clique = self.make_clique_maximal(graph, clique)
            if not self.is_maximal_clique(graph, clique):
                results[method_name] = {"clique": [], "score": 0}
            else:
                bt.logging.info(f"Found this clique using method: {method_name}, size: {len(clique)}")    
                results[method_name] = {"clique": sorted(clique), "score": len(clique)}
        
        # Find best valid clique
        sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        
        for method_name, result in sorted_results:
            clique = result["clique"]
            bt.logging.info(f"Found this clique using method: {method_name}, size: {len(clique)}")
            # if self.is_valid_clique(graph, clique):
            #     bt.logging.info(f"Found valid clique using method: {method_name}, size: {len(clique)}")
            if self.is_maximal_clique(graph, clique):
                bt.logging.info(f"Found maximal clique using method: {method_name}, size: {len(clique)}")
                return {
                    "winner": clique,
                    "method": method_name,
                    "all_results": results,
                }
        
        bt.logging.warning("No valid clique found, returning single node")
        single_node = [list(graph.nodes())[0]] if graph.nodes() else []
        return {
            "winner": single_node,
            "method": "fallback_single_node",
            "all_results": results,
        }

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        bt.logging.info(f"Received Synapse: {synapse.number_of_nodes} {synapse.timeout}")
        
        number_of_nodes = synapse.number_of_nodes
        adjacency_list = synapse.adjacency_list
        dict_of_lists = {i: adjacency_list[i] for i in range(number_of_nodes)}
        graph = nx.from_dict_of_lists(dict_of_lists)
        
        num_nodes = graph.number_of_nodes()
        
        result = self.find_maximum_clique_hybrid(graph)
        winner = result["winner"]
        method = result["method"]
            
        bt.logging.info(f"Winner method: {method}, clique size: {len(winner)}")

        synapse.adjacency_list = [[]]
        synapse.maximum_clique = sorted(winner)
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
