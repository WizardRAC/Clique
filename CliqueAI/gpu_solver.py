#!/usr/bin/env python3
"""
gpu_max_clique_30s.py

Parallel GPU multi-start greedy maximum-clique search.
- Uses PyTorch/CUDA
- Splits work across multiple GPUs (one subprocess per GPU)
- Runs for TIME_LIMIT seconds and returns the best clique found so far
"""

import time
import random
import argparse
import multiprocessing as mp
import os
import sys

try:
    import torch
except Exception as e:
    print("This script requires PyTorch with CUDA. Install PyTorch with CUDA and try again.")
    raise

# -----------------------
# Configurable params
# -----------------------
SEED = 44
TIME_LIMIT = 3.0  # seconds total run time
BATCH_PER_GPU = 2048  # number of parallel starts processed at once per GPU
VERBOSE = True
# -----------------------


def worker_loop(device_id, adj_matrix, batch_size, time_limit, shared_top, seed_offset):
    """
    Worker process function to run on a single GPU device.
    
    Args:
        device_id: GPU device ID
        adj_matrix: Adjacency matrix as numpy array or torch tensor (will be converted to GPU)
        batch_size: Number of parallel greedy searches per batch
        time_limit: Time limit in seconds
        shared_top: Shared dict for storing top results
        seed_offset: Random seed offset for this worker
    """
    import torch
    import numpy as np
    
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    if isinstance(adj_matrix, np.ndarray):
        adj = torch.from_numpy(adj_matrix).to(device=device, dtype=torch.bool)
    elif isinstance(adj_matrix, torch.Tensor):
        adj = adj_matrix.to(device=device, dtype=torch.bool)
    else:
        # Assume it's a list of lists
        adj = torch.tensor(adj_matrix, device=device, dtype=torch.bool)
    
    n = adj.shape[0]
    
    print(f"[GPU {device_id}] Starting with {n} nodes on device")
    
    rng = torch.Generator(device=device)
    rng.manual_seed(SEED + seed_offset + device_id)

    end_time = time.time() + time_limit
    batch_count = 0

    while time.time() < end_time:
        batch_count += 1
        B = batch_size
        clique_masks = torch.zeros((B, n), dtype=torch.bool, device=device)
        start_nodes = torch.randint(0, n, (B,), generator=rng, device=device)
        clique_masks[torch.arange(B, device=device), start_nodes] = True
        candidate = adj[start_nodes] & (~clique_masks)

        # Greedy expansion loop
        while candidate.any():
            cand_counts = (candidate.unsqueeze(2) & adj.unsqueeze(0)).sum(dim=2)
            cand_counts = cand_counts * candidate.to(torch.int32)
            picks = cand_counts.argmax(dim=1)
            clique_masks[torch.arange(B, device=device), picks] = True
            candidate = candidate & adj[picks] & (~clique_masks)

        sizes = clique_masks.sum(dim=1)
        # push all batch results into top list
        with shared_top['lock']:
            for i in range(B):
                size_i = int(sizes[i].item())
                if size_i > 0:
                    nodes_i = clique_masks[i].nonzero(as_tuple=False).squeeze(1).tolist()
                    shared_top['list'].append((size_i, nodes_i))
            # keep top 50 only
            current_list = list(shared_top['list'])
            if len(current_list) > 50:
                sorted_list = sorted(current_list, key=lambda x: x[0], reverse=True)[:50]
                shared_top['list'][:] = sorted_list

    if VERBOSE:
        print(f"[GPU {device_id}] finished {batch_count} batches.")


def run_multi_gpu(adj_matrix, time_limit=TIME_LIMIT, batch_per_gpu=BATCH_PER_GPU):
    """
    Run multi-GPU clique search on provided adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix (numpy array, torch tensor, or list of lists)
        time_limit: Time limit in seconds
        batch_per_gpu: Batch size per GPU
        
    Returns:
        Tuple of (top_results, elapsed_time)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script requires at least one CUDA GPU.")
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found.")
    if VERBOSE:
        print(f"CUDA devices found: {num_gpus}")
        for i in range(num_gpus):
            print(" -", torch.cuda.get_device_name(i))

    manager = mp.Manager()
    shared_top = manager.dict()
    shared_top['list'] = manager.list()
    shared_top['lock'] = manager.Lock()

    procs = []
    start_time = time.time()
    for dev in range(num_gpus):
        p = mp.Process(target=worker_loop, args=(dev, adj_matrix, batch_per_gpu, time_limit, shared_top, dev * 1000))
        p.daemon = True
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=time_limit + 7.0)

    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join()

    elapsed = time.time() - start_time
    top_results = sorted(list(shared_top['list']), key=lambda x: x[0], reverse=True)[:50]
    return top_results, elapsed

if __name__ == "__main__":
    import numpy as np
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description="Multi-GPU parallel greedy clique search.")
    parser.add_argument("--n", type=int, default=500, help="Number of nodes for test graph")
    parser.add_argument("--p", type=float, default=0.99, help="Edge probability for test graph")
    parser.add_argument("--time", type=float, default=TIME_LIMIT, help="time limit in seconds")
    parser.add_argument("--batch", type=int, default=BATCH_PER_GPU, help="batch size per GPU")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # Generate test graph
    print(f"Generating random test graph with {args.n} nodes and edge probability {args.p}")
    rng = np.random.RandomState(args.seed)
    arr = rng.rand(args.n, args.n)
    arr = np.triu(arr, 1)
    arr = arr + arr.T
    adj_matrix = (arr < args.p).astype(np.uint8)
    
    print("Parameters:")
    print("  N =", args.n)
    print("  EDGE_PROB =", args.p)
    print("  TIME_LIMIT =", args.time)
    print("  BATCH_PER_GPU =", args.batch)
    
    top_results, elapsed = run_multi_gpu(adj_matrix, time_limit=args.time, batch_per_gpu=args.batch)

    print(f"\nElapsed wall time: {elapsed:.3f}s")
    print(f"\nTOP {len(top_results)} CLIQUES FOUND:")
    for rank, (size, nodes) in enumerate(top_results[:20], 1):
        print(f"{rank:02d}: size={size}, nodes={nodes[:10]}{'...' if len(nodes) > 10 else ''}")
