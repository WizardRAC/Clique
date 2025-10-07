"""
Worker service that performs GPU-accelerated maximum clique finding.
Miners request work from this worker instead of computing locally.
Each miner receives ONE result per request.
"""

import time
import hashlib
import json
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import torch
import multiprocessing as mp
import traceback
import logging
import sys
import threading
import numpy as np

from CliqueAI.gpu_solver import run_multi_gpu

class TaskCache:
    """
    Cache for storing computed tasks with TTL of ~1 minute.
    Prevents duplicate processing and stores multiple good solutions.
    Tracks which solutions have been distributed to avoid duplicates.
    """
    
    def __init__(self, ttl_seconds: int = 60):
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        
    def _compute_hash(self, adjacency_list: List[List[int]], number_of_nodes: int) -> str:
        """Compute unique hash for a graph problem."""
        data = {
            "nodes": number_of_nodes,
            "edges": sorted([sorted(edge) for edge in adjacency_list if edge])
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def get_next_solution(self, adjacency_list: List[List[int]], number_of_nodes: int) -> Optional[List[int]]:
        """
        Get next available solution for this problem.
        Returns ONE solution and marks it as used.
        Different miners get different good solutions (max, max-1, max-2).
        """
        task_hash = self._compute_hash(adjacency_list, number_of_nodes)
        
        if task_hash in self.cache:
            entry = self.cache[task_hash]
            
            # Check if expired
            if time.time() - entry['timestamp'] >= self.ttl_seconds:
                del self.cache[task_hash]
                return None
            
            # Get next unused solution
            solutions = entry['solutions']
            used_indices = entry['used_indices']
            
            # Find next unused solution
            for i, (size, nodes) in enumerate(solutions):
                if i not in used_indices:
                    used_indices.add(i)
                    self.cache.move_to_end(task_hash)
                    return nodes
            
            # All solutions used, start over with best solution
            entry['used_indices'] = {0}
            return solutions[0][1] if solutions else None
        
        return None
    
    def set(self, adjacency_list: List[List[int]], number_of_nodes: int, 
            solutions: List[Tuple[int, List[int]]]):
        """
        Store computed solutions in cache.
        Filters to keep only good solutions (max, max-1, max-2 node sizes).
        """
        task_hash = self._compute_hash(adjacency_list, number_of_nodes)
        
        if solutions:
            max_size = solutions[0][0]
            good_solutions = [
                (size, nodes) for size, nodes in solutions 
                if size >= max_size - 2
            ]
        else:
            good_solutions = []
        
        self.cache[task_hash] = {
            'timestamp': time.time(),
            'solutions': good_solutions,
            'used_indices': set()
        }
        self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['timestamp'] >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]


class CliqueWorker:
    """
    GPU-accelerated worker that computes maximum cliques.
    Miners request work from this worker.
    Each miner receives ONE solution per request.
    """
    
    def __init__(self, time_limit: float = 3.0, batch_per_gpu: int = 2048):
        self.time_limit = time_limit
        self.batch_per_gpu = batch_per_gpu
        self.cache = TaskCache(ttl_seconds=60)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Worker requires GPU.")
        
        self.num_gpus = torch.cuda.device_count()
        print(f"Worker initialized with {self.num_gpus} GPU(s)")
        
        self._install_exception_hook()

    def _install_exception_hook(self):
        """Install a threading exception hook to suppress EOFError from logging monitoring threads."""
        original_hook = threading.excepthook
        
        def custom_excepthook(args):
            # Suppress EOFError from logging queue monitoring threads
            if (args.exc_type == EOFError and 
                args.thread and 
                args.thread.name and 
                '_monitor' in args.thread.name):
                # This is the expected EOFError from logging queue when child processes exit
                # Silently ignore it
                return
            # For all other exceptions, use the original hook
            if original_hook:
                original_hook(args)
        
        threading.excepthook = custom_excepthook

    def compute_clique(self, adjacency_list: List[List[int]], 
                      number_of_nodes: int) -> List[int]:
        """
        Compute maximum clique for the given graph.
        Returns ONE solution per request.
        Uses cache to avoid recomputation and provides different solutions to different miners.
        
        Args:
            adjacency_list: List of neighbor lists for each node
            number_of_nodes: Total number of nodes in the graph
            
        Returns:
            List of node indices forming a maximum (or near-maximum) clique
        """
        # Check cache first
        cached_solution = self.cache.get_next_solution(adjacency_list, number_of_nodes)
        
        if cached_solution is not None:
            print(f"[Worker] Returning cached solution of size {len(cached_solution)}")
            return cached_solution
        
        # Not in cache, compute new solutions
        print(f"[Worker] Computing new solutions for graph with {number_of_nodes} nodes...")
        solutions = self._run_gpu_search(adjacency_list, number_of_nodes)
        
        if not solutions:
            print(f"[Worker] WARNING: No solution found, returning empty clique")
            return []
        
        # Store in cache
        self.cache.set(adjacency_list, number_of_nodes, solutions)
        
        # Return the best solution for this first request
        best_solution = solutions[0][1]
        print(f"[Worker] Returning new solution of size {len(best_solution)}")
        
        # Mark this solution as used
        self.cache.get_next_solution(adjacency_list, number_of_nodes)
        
        return best_solution
    
    def _run_gpu_search(self, adjacency_list: List[List[int]], 
                       number_of_nodes: int) -> List[Tuple[int, List[int]]]:
        """
        Run GPU-accelerated maximum clique search using the gpu_solver module.
        Returns top solutions including max, max-1, and max-2 sizes.
        """
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set
        
        print(f"[Worker] Building adjacency matrix for {number_of_nodes} nodes...")
        adj_matrix = np.zeros((number_of_nodes, number_of_nodes), dtype=np.uint8)
        
        edge_count = 0
        for node, neighbors in enumerate(adjacency_list):
            if neighbors and node < number_of_nodes:
                for neighbor in neighbors:
                    if 0 <= neighbor < number_of_nodes and neighbor != node:
                        adj_matrix[node, neighbor] = 1
                        adj_matrix[neighbor, node] = 1
                        edge_count += 1
        
        edge_count = edge_count // 2
        print(f"[Worker] Graph has {edge_count} edges")
        
        if edge_count == 0:
            print(f"[Worker] WARNING: Graph has no edges!")
            return []
        
        try:
            print(f"[Worker] Starting GPU search with {self.num_gpus} GPUs for {self.time_limit}s...")
            start_time = time.time()
            
            top_results, elapsed = run_multi_gpu(
                adj_matrix=adj_matrix,
                time_limit=self.time_limit,
                batch_per_gpu=self.batch_per_gpu
            )
            
            print(f"[Worker] GPU search completed in {elapsed:.2f}s")
            print(f"[Worker] Collected {len(top_results)} results")
            
            if not top_results:
                print(f"[Worker] WARNING: No results collected from GPU processes")
                return []
            
            # Ensure all data is JSON-serializable (plain Python types)
            clean_results = [(int(size), [int(n) for n in nodes]) for size, nodes in top_results]
            
            print(f"[Worker] Best solution size: {clean_results[0][0] if clean_results else 0}")
            return clean_results
            
        except Exception as e:
            print(f"[Worker] Error in GPU search: {e}")
            traceback.print_exc()
            return []

_worker_instance: Optional[CliqueWorker] = None


def get_worker() -> CliqueWorker:
    """Get or create the global worker instance."""
    global _worker_instance
    
    if _worker_instance is None:
        _worker_instance = CliqueWorker(time_limit=3.0, batch_per_gpu=2048)
    return _worker_instance


if __name__ == "__main__":
    print("Testing CliqueWorker...")
    
    # Create a simple test graph
    n = 100
    adjacency_list = [list(range(i+1, min(i+10, n))) for i in range(n)]
    
    worker = get_worker()
    
    print(f"\nComputing clique for graph with {n} nodes...")
    start = time.time()
    clique1 = worker.compute_clique(adjacency_list, n)
    elapsed = time.time() - start
    print(f"Miner 1 received clique of size {len(clique1)} in {elapsed:.2f}s")
    
    print("\nSecond miner requesting same problem (should use cache)...")
    start = time.time()
    clique2 = worker.compute_clique(adjacency_list, n)
    elapsed = time.time() - start
    print(f"Miner 2 received clique of size {len(clique2)} in {elapsed:.4f}s")
    print(f"Solutions are different: {clique1 != clique2}")
    
    print("\nThird miner requesting same problem...")
    start = time.time()
    clique3 = worker.compute_clique(adjacency_list, n)
    elapsed = time.time() - start
    print(f"Miner 3 received clique of size {len(clique3)} in {elapsed:.4f}s")
