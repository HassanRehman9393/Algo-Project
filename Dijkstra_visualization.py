"""
Visualization for Bitcoin Alpha Trust Network
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np

def visualize_graph(data_file, limit=None):
    """Visualize the Bitcoin Alpha trust network graph"""
    # Create output directory if it doesn't exist
    os.makedirs("visualization", exist_ok=True)
    
    print("Loading graph data for visualization...")
    G = nx.DiGraph()
    
    # Track observed nodes for limiting
    observed_nodes = set()
    
    with open(data_file, 'r') as file:
        # Skip comment line if present
        first_line = file.readline()
        if first_line.startswith('//'):
            pass  # Skip comment line
        else:
            # Reset file pointer if first line was not a comment
            file.seek(0)
        
        for line in file:
            if not line.strip() or line.startswith('//'):
                continue
            
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            
            source, target, rating, _ = parts
            source, target = int(source), int(target)
            rating = int(rating)
            
            # Track unique nodes for limiting
            observed_nodes.add(source)
            observed_nodes.add(target)
            
            # Check if we've reached our node limit
            if limit and len(observed_nodes) > limit:
                break
            
            # Add edge with rating as weight
            G.add_edge(source, target, weight=rating)
    
    # Calculate basic graph metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Calculate trust distribution
    trust_values = [d['weight'] for _, _, d in G.edges(data=True)]
    positive_edges = sum(1 for w in trust_values if w > 0)
    negative_edges = sum(1 for w in trust_values if w < 0)
    neutral_edges = sum(1 for w in trust_values if w == 0)
    
    print(f"Graph loaded with {num_nodes} nodes and {num_edges} edges")
    print(f"Trust distribution: {positive_edges} positive, {negative_edges} negative, {neutral_edges} neutral")
    
    # Limit visualization to a smaller subset if graph is too large
    if limit is None and num_nodes > 1000:
        print("Graph is very large, limiting visualization to 500 nodes...")
        # Create a subgraph with the most connected nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
        G = G.subgraph(top_nodes).copy()
    
    # Use spring layout for cleaner visualization
    print("Calculating graph layout (this may take a while)...")
    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), iterations=50)
    
    # Set up colors based on trust level (edge weight)
    # Red for negative trust, green for positive trust
    edge_colors = [determine_edge_color(d['weight']) for _, _, d in G.edges(data=True)]
    
    # Adjust edge widths based on absolute trust value
    edge_widths = [abs(d['weight'])/3 + 0.5 for _, _, d in G.edges(data=True)]
    
    # Set up node sizes based on degree centrality
    node_sizes = [calculate_node_size(G, node) for node in G.nodes()]
    
    # Create figure
    print("Generating visualization...")
    plt.figure(figsize=(20, 20))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                           alpha=0.7, arrows=True, arrowsize=10, 
                           connectionstyle='arc3,rad=0.1')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', 
                           alpha=0.8, edgecolors='black')
    
    # Optional: Draw labels for most important nodes
    if num_nodes <= 100:  # Only show labels for small graphs
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    elif num_nodes <= 500:  # For medium-sized graphs, show top node labels
        degrees = dict(G.degree())
        top_nodes = [node for node, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:50]]
        labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Add a color bar to indicate trust levels
    create_trust_colorbar(plt)
    
    # Add title and axis labels
    plt.title(f'Bitcoin Alpha Trust Network\n{num_nodes} nodes, {num_edges} edges', fontsize=16)
    plt.axis('off')  # Hide axes
    
    # Save the visualization
    output_path = os.path.join("visualization", f"bitcoin_alpha_trust_network{'_limited' if limit else ''}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Create additional visualization focusing on trust distribution
    create_trust_distribution_plot(trust_values)

def determine_edge_color(weight):
    """Determine color based on trust weight"""
    norm = mcolors.Normalize(vmin=-10, vmax=10)
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    return cmap(norm(weight))

def calculate_node_size(graph, node):
    """Calculate node size based on degree centrality"""
    base_size = 50
    degree = graph.degree(node)
    # Scale size logarithmically to prevent extremely large nodes
    return base_size + 30 * np.log1p(degree)

def create_trust_colorbar(plt):
    """Add a colorbar to indicate trust levels"""
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-10, vmax=10)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.01, fraction=0.05,
                        label='Trust Level (-10 to +10)')
    cbar.set_ticks(range(-10, 11, 2))

def create_trust_distribution_plot(trust_values):
    """Create a histogram showing the distribution of trust values"""
    plt.figure(figsize=(12, 8))
    
    bins = range(-11, 12)  # -10 to +10 inclusive
    plt.hist(trust_values, bins=bins, edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Trust Values in Bitcoin Alpha Network', fontsize=16)
    plt.xlabel('Trust Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(range(-10, 11))
    
    # Save the visualization
    output_path = os.path.join("visualization", "trust_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trust distribution plot saved to {output_path}")

if __name__ == "__main__":
    data_file = "c:\\Users\\Hp\\Desktop\\Algo-Project\\soc-sign-bitcoinalpha.csv"
    visualize_graph(data_file)
