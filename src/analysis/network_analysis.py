"""
Network analysis tools for Synthia.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
import networkx as nx


class NetworkAnalyzer:
    """
    Analyze biological networks (GRN, metabolic, signaling).
    """
    
    def __init__(self, graph: nx.Graph = None):
        self.graph = graph or nx.DiGraph()
    
    def set_graph(self, graph: nx.Graph):
        """Set the network graph."""
        self.graph = graph
    
    def add_edge(self, source: str, target: str, **attrs):
        """Add edge to network."""
        self.graph.add_edge(source, target, **attrs)
    
    def add_node(self, node: str, **attrs):
        """Add node to network."""
        self.graph.add_node(node, **attrs)
    
    @staticmethod
    def calculate_degree_distribution(graph: nx.Graph) -> Dict[str, float]:
        """Calculate degree distribution statistics."""
        degrees = [d for n, d in graph.degree()]
        
        if not degrees:
            return {}
        
        return {
            'mean_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'degree_std': np.std(degrees),
            'total_nodes': len(degrees),
            'total_edges': graph.number_of_edges()
        }
    
    @staticmethod
    def calculate_clustering_coefficient(graph: nx.Graph) -> float:
        """Calculate average clustering coefficient."""
        if isinstance(graph, nx.DiGraph):
            # Convert to undirected for clustering
            graph = graph.to_undirected()
        
        return nx.average_clustering(graph)
    
    @staticmethod
    def find_shortest_path(graph: nx.Graph, source: str, target: str) -> Tuple[List[str], float]:
        """Find shortest path between nodes."""
        try:
            path = nx.shortest_path(graph, source, target)
            length = nx.shortest_path_length(graph, source, target)
            return path, length
        except nx.NetworkXNoPath:
            return [], float('inf')
    
    @staticmethod
    def identify_hub_nodes(graph: nx.Graph, top_k: int = 10) -> List[Tuple[str, int]]:
        """Identify hub nodes by degree."""
        degrees = [(n, d) for n, d in graph.degree()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_k]
    
    @staticmethod
    def find_communities(graph: nx.Graph) -> List[Set[str]]:
        """Find network communities."""
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(graph)
            return list(communities)
        except:
            # Fallback to connected components
            return list(nx.connected_components(graph.to_undirected()))
    
    @staticmethod
    def calculate_centrality(graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures."""
        centrality = {
            'betweenness': nx.betweenness_centrality(graph),
            'closeness': nx.closeness_centrality(graph),
            'degree': nx.degree_centrality(graph),
        }
        
        try:
            centrality['eigenvector'] = nx.eigenvector_centrality(graph)
        except:
            pass
        
        return centrality
    
    @staticmethod
    def find_feedback_loops(graph: nx.Graph) -> List[List[str]]:
        """Find feedback loops (cycles) in the network."""
        try:
            return list(nx.simple_cycles(graph))
        except:
            return []
    
    @staticmethod
    def calculate_network_resilience(graph: nx.Graph) -> Dict[str, float]:
        """Calculate network resilience metrics."""
        if isinstance(graph, nx.DiGraph):
            graph = graph.to_undirected()
        
        # Degree assortativity
        assortativity = nx.degree_assortativity_coefficient(graph)
        
        # Percolation threshold (approximate)
        n = graph.number_of_nodes()
        avg_degree = sum(d for n, d in graph.degree()) / n if n > 0 else 0
        
        percolation_threshold = 1 - 1/avg_degree if avg_degree > 1 else 0
        
        return {
            'assortativity': assortativity,
            'percolation_threshold': percolation_threshold,
            'avg_degree': avg_degree
        }
    
    @staticmethod
    def identify_bottlenecks(graph: nx.Graph) -> List[str]:
        """Identify bottleneck nodes (high betweenness centrality)."""
        betweenness = nx.betweenness_centrality(graph)
        threshold = np.mean(list(betweenness.values())) + np.std(list(betweenness.values()))
        
        return [node for node, bc in betweenness.items() if bc > threshold]
    
    @staticmethod
    def calculate_network_entropy(graph: nx.Graph) -> float:
        """Calculate network entropy (complexity measure)."""
        degrees = [d for n, d in graph.degree()]
        total = sum(degrees)
        
        if total == 0:
            return 0.0
        
        # Normalize degrees
        probs = [d/total for d in degrees]
        
        # Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def motif_detection(graph: nx.Graph, motif_size: int = 3) -> Dict[Tuple, int]:
        """Count network motifs (subgraphs of size n)."""
        # This is a simplified version
        # Real motif detection uses comparison with random networks
        
        if motif_size != 3:
            return {}
        
        motifs = {
            ('ffl',): 0,  # Feed-forward loop
            ('bifan',): 0,  # Bifan
            ('loop',): 0,  # Feedback loop
        }
        
        # Count simple 3-node motifs
        for node in graph.nodes():
            neighbors = list(graph.successors(node))
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2:
                        # Check for connections
                        if graph.has_edge(n1, n2):
                            motifs[('bifan',)] += 1
        
        # Count loops
        for node in graph.nodes():
            if graph.has_edge(node, node):
                motifs[('loop',)] += 1
        
        return motifs
    
    def get_network_summary(self) -> Dict:
        """Get comprehensive network summary."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'degree_distribution': self.calculate_degree_distribution(self.graph),
            'clustering': self.calculate_clustering_coefficient(self.graph),
            'centralities': self.calculate_centrality(self.graph),
            'hub_nodes': self.identify_hub_nodes(self.graph, 5),
            'entropy': self.calculate_network_entropy(self.graph),
            'resilience': self.calculate_network_resilience(self.graph)
        }
