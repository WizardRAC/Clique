import torch
import time
import random
import collections
from typing import List, Dict, Tuple
import bittensor as bt


class GPUCliqueFinder:
    """
    GPU-accelerated parallel clique finder using PyTorch/CUDA.
    Runs multiple greedy searches in parallel on GPU.
    """
    
    def __init__(self, device=None):
        """Initialize GPU clique finder with specified device."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if self.device.type == 'cuda':
            bt.logging.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            bt.logging.warning("CUDA not available, falling back to CPU")
    
    def graph_to_adjacency_matrix(self, graph) -> torch.Tensor:
        """Convert NetworkX graph to GPU adjacency matrix."""
        num_nodes = graph.number_of_nodes()
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=self.device)
        
        # Build adjacency matrix
        for u, v in graph.edges():
            adj_matrix[u, v] = True
            adj_matrix[v, u] = True
        
        return adj_matrix
    
    def parallel_greedy_search(
        self, 
        adj_matrix: torch.Tensor, 
        num_searches: int = 200,
        timeout: float = 17.0,
        seed: int = None
    ) -> Dict:
        """
        Run multiple greedy clique searches in parallel on GPU.
        
        Args:
            adj_matrix: Boolean adjacency matrix on GPU
            num_searches: Number of parallel searches to run (auto-adjusted based on graph size)
            timeout: Maximum time in seconds
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary with best clique and statistics
        """
        start_time = time.time()
        num_nodes = adj_matrix.shape[0]
        
        # Smaller graphs (100-200 nodes) can handle more iterations
        if 100 <= num_nodes <= 200:
            num_searches = 400
            bt.logging.info(f"Graph size {num_nodes} nodes: using {num_searches} iterations")
        elif num_nodes < 100:
            num_searches = 600  # Even smaller graphs can do more
            bt.logging.info(f"Small graph ({num_nodes} nodes): using {num_searches} iterations")
        else:
            bt.logging.info(f"Large graph ({num_nodes} nodes): using {num_searches} iterations")
        
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Precompute degree for each node
        degrees = adj_matrix.sum(dim=1)
        
        # Store results
        all_cliques = []
        clique_sizes = []
        uniq_counts = {}
        
        # Run searches in batches to manage memory
        batch_size = min(50, num_searches)
        num_batches = (num_searches + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            if time.time() - start_time >= timeout:
                bt.logging.warning(f"Timeout reached at batch {batch_idx}/{num_batches}")
                break
            
            current_batch_size = min(batch_size, num_searches - batch_idx * batch_size)
            
            # Run parallel greedy searches for this batch
            batch_cliques = self._batch_greedy_search(
                adj_matrix, 
                degrees, 
                current_batch_size,
                num_nodes
            )
            
            # Process results
            for clique in batch_cliques:
                clique_list = clique.tolist()
                clique_sizes.append(len(clique_list))
                all_cliques.append(clique_list)
                
                # Track unique cliques
                key = frozenset(clique_list)
                uniq_counts[key] = uniq_counts.get(key, 0) + 1
        
        # Find best clique (largest size, lowest frequency)
        if uniq_counts:
            best_clique, best_occ = max(
                uniq_counts.items(), 
                key=lambda kv: (len(kv[0]), -kv[1])
            )
            
            # Find rare-large clique (lowest frequency, largest size)
            rare_large, rl_occ = min(
                uniq_counts.items(), 
                key=lambda kv: (kv[1], -len(kv[0]))
            )
            
            bt.logging.info(f"GPU search completed: {len(all_cliques)} searches")
            bt.logging.info(f"Best clique size: {len(best_clique)}, occurrences: {best_occ}")
            bt.logging.info(f"Rare-large clique size: {len(rare_large)}, occurrences: {rl_occ}")
            
            # Choose between best and rare-large
            if len(rare_large) >= len(best_clique):
                final_clique = sorted(rare_large)
            elif rl_occ >= best_occ:
                final_clique = sorted(best_clique)
            else:
                final_clique = sorted(rare_large)
            
            return {
                "clique": final_clique,
                "score": len(final_clique),
                "num_searches": len(all_cliques),
                "unique_cliques": len(uniq_counts),
                "size_distribution": dict(collections.Counter(clique_sizes))
            }
        else:
            return {
                "clique": [],
                "score": 0,
                "num_searches": 0,
                "unique_cliques": 0,
                "size_distribution": {}
            }
    
    def _batch_greedy_search(
        self, 
        adj_matrix: torch.Tensor, 
        degrees: torch.Tensor,
        batch_size: int,
        num_nodes: int
    ) -> List[torch.Tensor]:
        """
        Run a batch of greedy searches in parallel on GPU.
        
        Each search starts from a random high-degree node and greedily expands.
        """
        cliques = []
        
        # Generate random starting nodes for each search
        # Bias towards high-degree nodes
        degree_probs = (degrees.float() + 1.0) ** 1.2
        degree_probs = degree_probs / degree_probs.sum()
        
        for _ in range(batch_size):
            # Sample starting node based on degree
            start_node = torch.multinomial(degree_probs, 1).item()
            
            # Greedy expansion
            current_clique = torch.tensor([start_node], device=self.device)
            
            # Find candidates: nodes connected to all nodes in current clique
            candidates_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
            candidates_mask[start_node] = False
            
            # Iteratively add nodes
            max_iterations = 100
            for _ in range(max_iterations):
                # Update candidates: must be connected to all nodes in clique
                for node in current_clique:
                    candidates_mask &= adj_matrix[node]
                
                # Find valid candidates
                valid_candidates = torch.where(candidates_mask)[0]
                
                if len(valid_candidates) == 0:
                    break
                
                # Choose candidate with highest degree among valid ones
                candidate_degrees = degrees[valid_candidates]
                best_idx = torch.argmax(candidate_degrees)
                best_candidate = valid_candidates[best_idx]
                
                # Add to clique
                current_clique = torch.cat([current_clique, best_candidate.unsqueeze(0)])
                candidates_mask[best_candidate] = False
            
            cliques.append(current_clique)
        
        return cliques
    
    def is_valid_clique(self, adj_matrix: torch.Tensor, clique: List[int]) -> bool:
        """Verify if a set of nodes forms a valid clique."""
        if len(clique) <= 1:
            return len(clique) == 1
        
        clique_tensor = torch.tensor(clique, device=self.device)
        
        # Check all pairs are connected
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                if not adj_matrix[clique[i], clique[j]]:
                    return False
        return True
