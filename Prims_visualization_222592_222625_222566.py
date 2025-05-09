"""
Visualization for Prim's Algorithm on Bitcoin Alpha Network
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
from Prims_222592_222625_222566 import Graph

def visualize_mst(data_file, start_vertex=None, limit=None):
    """Visualize the Minimum Spanning Tree of the Bitcoin Alpha network"""
    # Create output directory if it doesn't exist
    os.makedirs("visualization", exist_ok=True)
    
    print("Loading graph data for visualization...")
    
    # Create and load the graph using our Prim's implementation
    graph = Graph()
    graph.load_from_file(data_file, limit)
    
    # If no start vertex provided, use the first vertex in the graph
    if start_vertex is None or start_vertex not in graph.vertices:
        start_vertex = next(iter(graph.vertices))
    
    # Run Prim's algorithm
    trace_file = os.path.join("visualization", "temp_trace.txt")
    mst_edges, total_weight = graph.prims_mst(start_vertex, trace_file)
    
    # Create NetworkX graph for visualization
    G = nx.Graph()  # Undirected graph for MST
    
    # Add all vertices first
    for vertex in graph.vertices:
        G.add_node(vertex)
    
    # Add MST edges
    for edge in mst_edges:
        source, target, weight = edge
        G.add_edge(source, target, weight=abs(weight))
    
    # Calculate basic graph metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    print(f"MST contains {num_nodes} nodes and {num_edges} edges")
    print(f"Total MST weight: {total_weight}")
      # Limit visualization to a smaller subset if graph is too large
    if limit is None and num_nodes > 1000:
        print("Graph is very large, limiting visualization to 1000 nodes...")
        # Create a subgraph with the most connected nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:1000]
        # Make sure start_vertex is included
        if start_vertex not in top_nodes:
            top_nodes[0] = start_vertex
        G = G.subgraph(top_nodes).copy()
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
    
    # Use spring layout for cleaner visualization
    print("Calculating graph layout (this may take a while)...")
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=100, weight='weight')
    
    # Set up edge colors based on weight
    edge_colors = [determine_edge_color(d['weight']) for _, _, d in G.edges(data=True)]
    
    # Adjust edge widths based on weight
    edge_widths = [2 + abs(d['weight'])/2 for _, _, d in G.edges(data=True)]
    
    # Set up node sizes based on degree centrality
    node_sizes = [calculate_node_size(G, node) for node in G.nodes()]
    
    # Highlight start vertex
    node_colors = ['red' if node == start_vertex else 'skyblue' for node in G.nodes()]
      # Create figure with a specific size and layout for colorbar
    print("Generating visualization...")
    fig = plt.figure(figsize=(20, 20))
    
    # Create main axis for the graph
    ax_graph = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                          alpha=0.7, ax=ax_graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.8, edgecolors='black', ax=ax_graph)
    
    # Draw labels based on graph size
    if num_nodes <= 100:  # Show all labels for small graphs
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax_graph)
    elif num_nodes <= 500:  # For medium-sized graphs, show important node labels
        degrees = dict(G.degree())
        top_nodes = [node for node, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:50]]
        top_nodes.append(start_vertex)  # Always show start vertex
        labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax_graph)
    
    # Add a color bar to indicate edge weights
    create_weight_colorbar(fig, ax_graph)
    
    # Add title and axis labels
    ax_graph.set_title(f'Minimum Spanning Tree - Bitcoin Alpha Network\n'
                      f'Start Vertex: {start_vertex}, {num_nodes} nodes, {num_edges} edges\n'
                      f'Total MST Weight: {total_weight:.2f}', fontsize=16, pad=20)
    ax_graph.axis('off')  # Hide axes
    
    # Save the visualization
    output_path = os.path.join("visualization", f"bitcoin_alpha_mst_from_{start_vertex}{'_limited' if limit else ''}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"MST visualization saved to {output_path}")
    
    # Create additional visualizations
    create_weight_distribution_plot(G)
    create_path_lengths_plot(G)

def determine_edge_color(weight):
    """Determine color based on edge weight"""
    norm = mcolors.Normalize(vmin=0, vmax=21)  # Adjusted for our weight range
    cmap = plt.cm.viridis  # Using viridis colormap for weight visualization
    return cmap(norm(weight))

def calculate_node_size(graph, node):
    """Calculate node size based on degree centrality"""
    base_size = 100
    degree = graph.degree(node)
    # Scale size logarithmically to prevent extremely large nodes
    return base_size + 50 * np.log1p(degree)

def create_weight_colorbar(fig, ax):
    """Add a colorbar to indicate edge weights"""
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=21)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Create a new axes for the colorbar below the main plot
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Edge Weight', fontsize=12)
    cbar.set_ticks(range(0, 22, 3))

def create_weight_distribution_plot(G):
    """Create a histogram showing the distribution of edge weights in the MST"""
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    
    plt.figure(figsize=(12, 8))
    plt.hist(weights, bins=20, edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Edge Weights in Minimum Spanning Tree', fontsize=16)
    plt.xlabel('Edge Weight', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    
    # Save the visualization
    output_path = os.path.join("visualization", "mst_weight_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Weight distribution plot saved to {output_path}")

def create_path_lengths_plot(G):
    """Create a plot showing the distribution of path lengths in the MST"""
    # Calculate all shortest paths lengths
    path_lengths = []
    for source in G.nodes():
        for target in G.nodes():
            if source < target:  # Only count each path once
                try:
                    length = nx.shortest_path_length(G, source, target)
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    continue
    
    plt.figure(figsize=(12, 8))
    plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths) + 2),
             edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Path Lengths in Minimum Spanning Tree', fontsize=16)
    plt.xlabel('Path Length (Number of Edges)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    
    # Save the visualization
    output_path = os.path.join("visualization", "mst_path_lengths.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Path lengths distribution plot saved to {output_path}")

if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "soc-sign-bitcoinalpha.csv")
    visualize_mst(data_file)
