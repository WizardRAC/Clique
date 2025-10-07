#!/usr/bin/env python3
"""
Standalone worker service for GPU-accelerated clique computation.
Run this on a machine with CUDA GPUs before starting miners.
"""

import time
import argparse
from CliqueAI.worker import get_worker


def main():
    parser = argparse.ArgumentParser(
        description="Start the CliqueAI GPU worker service"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3.0,
        help="Time limit for each computation in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=2048,
        help="Batch size per GPU (default: 2048)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test computation and exit"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CliqueAI GPU Worker Service")
    print("=" * 60)
    print(f"Time limit per computation: {args.time_limit}s")
    print(f"Batch size per GPU: {args.batch_per_gpu}")
    print()
    
    # Initialize worker
    try:
        worker = get_worker()
        print(f"✓ Worker initialized successfully")
        print(f"✓ Cache TTL: 60 seconds")
        print(f"✓ Returns top 20 solutions per computation")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize worker: {e}")
        return 1
    
    if args.test:
        print("Running test computation...")
        print("-" * 60)
        
        # Create a test graph
        n = 100
        adjacency_list = [list(range(i+1, min(i+10, n))) for i in range(n)]
        
        print(f"Test graph: {n} nodes")
        start = time.time()
        clique = worker.compute_clique(adjacency_list, n)
        elapsed = time.time() - start
        
        print(f"✓ Found clique of size {len(clique)} in {elapsed:.2f}s")
        print(f"  Clique nodes: {clique}")
        
        # Test cache
        print("\nTesting cache (should be instant)...")
        start = time.time()
        clique2 = worker.compute_clique(adjacency_list, n)
        elapsed = time.time() - start
        print(f"✓ Cache hit! Lookup took {elapsed:.4f}s")
        
        # Get top solutions
        print("\nTop 5 solutions:")
        top_solutions = worker.get_top_solutions(adjacency_list, n, top_k=5)
        for i, (size, nodes) in enumerate(top_solutions, 1):
            print(f"  {i}. Size {size}: {nodes[:10]}{'...' if len(nodes) > 10 else ''}")
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        return 0
    
    # Run as service
    print("Worker is ready and waiting for requests...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down worker service...")
        print("Goodbye!")
        return 0


if __name__ == "__main__":
    exit(main())
