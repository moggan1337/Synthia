"""
Visualization plots for Synthia.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def plot_time_series(time_points: List[float],
                    data: Dict[str, List[float]],
                    title: str = "Time Series",
                    xlabel: str = "Time",
                    ylabel: str = "Concentration",
                    save_path: Optional[str] = None):
    """
    Plot time series data.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, values in data.items():
        ax.plot(time_points, values, label=name, linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_network(network_data: Dict,
                title: str = "Network",
                save_path: Optional[str] = None):
    """
    Plot network graph.
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        print("Matplotlib or NetworkX not available")
        return
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in network_data.get('nodes', []):
        G.add_node(node)
    
    # Add edges
    for edge in network_data.get('edges', []):
        if len(edge) >= 2:
            G.add_edge(edge[0], edge[1])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                          node_size=500)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, 
                          arrowstyle='->', connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_sequence_logo(sequence_data: Dict[str, float],
                      title: str = "Sequence Logo",
                      save_path: Optional[str] = None):
    """
    Plot sequence logo (simplified).
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    positions = list(sequence_data.keys())
    heights = list(sequence_data.values())
    
    colors = {'A': 'red', 'T': 'blue', 'G': 'orange', 'C': 'green'}
    
    for i, (pos, height) in enumerate(zip(positions, heights)):
        color = colors.get(pos, 'gray')
        ax.bar(i, height, color=color, width=0.8)
    
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, max(heights) * 1.1 if heights else 1)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions, fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_pathway(steps: List[str],
                rates: List[float],
                title: str = "Metabolic Pathway",
                save_path: Optional[str] = None):
    """
    Plot metabolic pathway as bar chart.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(steps))
    
    ax.bar(x_pos, rates, color='steelblue', alpha=0.7)
    
    ax.set_xlabel("Reaction Step", fontsize=12)
    ax.set_ylabel("Rate (mM/s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_cell_cycle(phases: Dict[str, float],
                   title: str = "Cell Cycle Distribution",
                   save_path: Optional[str] = None):
    """
    Plot cell cycle phase distribution as pie chart.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(phases.keys())
    sizes = list(phases.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
          startangle=90, explode=[0.05] * len(sizes))
    
    ax.set_title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_population_dynamics(time_points: List[float],
                            populations: Dict[str, List[float]],
                            title: str = "Population Dynamics",
                            save_path: Optional[str] = None):
    """
    Plot population dynamics over time.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot populations
    for name, pop in populations.items():
        ax1.plot(time_points, pop, label=name, linewidth=2)
    
    ax1.set_xlabel("Time (hours)", fontsize=12)
    ax1.set_ylabel("Population", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot growth rates
    for name, pop in populations.items():
        pop_array = np.array(pop)
        growth_rates = np.gradient(np.log(pop_array + 1))
        ax2.plot(time_points[1:], growth_rates, label=f"{name} growth rate", 
                linewidth=2)
    
    ax2.set_xlabel("Time (hours)", fontsize=12)
    ax2.set_ylabel("Growth Rate (1/hour)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_heatmap(data: np.ndarray,
                x_labels: List[str] = None,
                y_labels: List[str] = None,
                title: str = "Heatmap",
                save_path: Optional[str] = None):
    """
    Plot heatmap.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    
    if x_labels:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(y_labels)
    
    ax.set_title(title, fontsize=14)
    
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_boxplot(data: Dict[str, List[float]],
                title: str = "Distribution Comparison",
                ylabel: str = "Value",
                save_path: Optional[str] = None):
    """
    Plot boxplot comparison.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(data.keys())
    values = list(data.values())
    
    bp = ax.boxplot(values, labels=labels, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
