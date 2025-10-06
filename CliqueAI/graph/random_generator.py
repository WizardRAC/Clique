import random
import uuid
from CliqueAI.graph.model import LambdaGraph


def generate_random_graph(
    label: str,
    number_of_nodes_min: int,
    number_of_nodes_max: int,
    number_of_edges_min: int,
    number_of_edges_max: int,
    edge_probability: float = 0.15,
) -> LambdaGraph:
    """
    Generate a random graph for testing purposes without needing Lambda service.
    
    Args:
        label: Problem type label
        number_of_nodes_min: Minimum number of nodes
        number_of_nodes_max: Maximum number of nodes
        number_of_edges_min: Minimum number of edges (not strictly enforced)
        number_of_edges_max: Maximum number of edges (not strictly enforced)
        edge_probability: Probability of edge between any two nodes (default 0.15)
    
    Returns:
        LambdaGraph object with random graph structure
    """
    # Generate random number of nodes within range
    number_of_nodes = random.randint(number_of_nodes_min, number_of_nodes_max)
    
    # Initialize adjacency list
    adjacency_list = [[] for _ in range(number_of_nodes)]
    
    # Generate random edges using Erdős–Rényi model
    edge_count = 0
    for i in range(number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            if random.random() < edge_probability:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
                edge_count += 1
    
    # Sort adjacency lists for consistency
    for i in range(number_of_nodes):
        adjacency_list[i].sort()
    
    # Generate unique UUID for this graph
    graph_uuid = str(uuid.uuid4())
    
    return LambdaGraph(
        uuid=graph_uuid,
        label=label,
        number_of_nodes=number_of_nodes,
        adjacency_list=adjacency_list,
    )


def generate_random_graph_with_planted_clique(
    label: str,
    number_of_nodes_min: int,
    number_of_nodes_max: int,
    clique_size: int = 10,
    edge_probability: float = 0.15,
) -> LambdaGraph:
    """
    Generate a random graph with a planted clique for more interesting test cases.
    
    Args:
        label: Problem type label
        number_of_nodes_min: Minimum number of nodes
        number_of_nodes_max: Maximum number of nodes
        clique_size: Size of the planted clique
        edge_probability: Probability of edge between non-clique nodes
    
    Returns:
        LambdaGraph object with planted clique
    """
    # Generate random number of nodes within range
    number_of_nodes = random.randint(number_of_nodes_min, number_of_nodes_max)
    
    # Ensure clique size is valid
    clique_size = min(clique_size, number_of_nodes)
    
    # Initialize adjacency list
    adjacency_list = [[] for _ in range(number_of_nodes)]
    
    # Select random nodes for the planted clique
    clique_nodes = random.sample(range(number_of_nodes), clique_size)
    
    # Create complete connections within the clique
    for i in range(len(clique_nodes)):
        for j in range(i + 1, len(clique_nodes)):
            node_i = clique_nodes[i]
            node_j = clique_nodes[j]
            adjacency_list[node_i].append(node_j)
            adjacency_list[node_j].append(node_i)
    
    # Add random edges between other nodes
    for i in range(number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            # Skip if both nodes are in the clique (already connected)
            if i in clique_nodes and j in clique_nodes:
                continue
            
            if random.random() < edge_probability:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
    
    # Sort adjacency lists for consistency
    for i in range(number_of_nodes):
        adjacency_list[i].sort()
    
    # Generate unique UUID for this graph
    graph_uuid = str(uuid.uuid4())
    
    return LambdaGraph(
        uuid=graph_uuid,
        label=label,
        number_of_nodes=number_of_nodes,
        adjacency_list=adjacency_list,
    )
