"""
Bellman-Ford Algorithm Visualization for CS-2009 Design and Analysis of Algorithms Project
Group: 222592, 222625, 222566
"""
import os
import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import pandas as pd
import seaborn as sns

def create_output_dir():
    """Create output directory for visualizations"""
    os.makedirs("visualization", exist_ok=True)

def load_algorithm_performance_data():
    """Load performance data from CSV files"""
    # Define file paths
    bellman_ford_csv = os.path.join("performance", "bellman_ford_performance.csv")
    dijkstra_csv = os.path.join("performance", "dijkstra_performance.csv")
    
    # Check if files exist
    if not os.path.exists(bellman_ford_csv) or not os.path.exists(dijkstra_csv):
        print("Performance data files not found. Please run performance analysis first.")
        return None, None
    
    # Load data
    bellman_ford_data = pd.read_csv(bellman_ford_csv)
    dijkstra_data = pd.read_csv(dijkstra_csv)
    
    return bellman_ford_data, dijkstra_data

def visualize_algorithm_comparison():
    """Create comparative visualizations between Bellman-Ford and Dijkstra algorithms"""
    # Load data
    bellman_ford_data, dijkstra_data = load_algorithm_performance_data()
    if bellman_ford_data is None or dijkstra_data is None:
        return
    
    create_output_dir()
    
    # Ensure we're comparing the same node sizes
    common_nodes = set(bellman_ford_data['nodes']).intersection(set(dijkstra_data['nodes']))
    
    if not common_nodes:
        print("No common node sizes found between the two algorithms.")
        return
    
    common_nodes = sorted(list(common_nodes))
    
    # Filter data for common node sizes
    bf_filtered = bellman_ford_data[bellman_ford_data['nodes'].isin(common_nodes)]
    dj_filtered = dijkstra_data[dijkstra_data['nodes'].isin(common_nodes)]
    
    # Create figure for comparison
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Execution Time Comparison
    plt.subplot(2, 2, 1)
    plt.plot(bf_filtered['nodes'], bf_filtered['execution_time'], 'o-', label='Bellman-Ford', linewidth=2, markersize=8)
    plt.plot(dj_filtered['nodes'], dj_filtered['execution_time'], 'o-', label='Dijkstra', linewidth=2, markersize=8)
    plt.title('Execution Time Comparison')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Time Ratio (Bellman-Ford / Dijkstra)
    plt.subplot(2, 2, 2)
    ratio = []
    for i, node in enumerate(common_nodes):
        bf_time = bf_filtered[bf_filtered['nodes'] == node]['execution_time'].values[0]
        dj_time = dj_filtered[dj_filtered['nodes'] == node]['execution_time'].values[0]
        ratio.append(bf_time / dj_time if dj_time > 0 else float('inf'))
    
    plt.bar(common_nodes, ratio)
    plt.title('Performance Ratio (Bellman-Ford / Dijkstra)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time Ratio')
    plt.grid(True, axis='y')
    
    # Plot 3: Log Scale Comparison
    plt.subplot(2, 2, 3)
    plt.plot(bf_filtered['nodes'], bf_filtered['execution_time'], 'o-', label='Bellman-Ford', linewidth=2, markersize=8)
    plt.plot(dj_filtered['nodes'], dj_filtered['execution_time'], 'o-', label='Dijkstra', linewidth=2, markersize=8)
    plt.title('Execution Time Comparison (Log Scale)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # Plot 4: Theoretical vs Actual Scaling
    plt.subplot(2, 2, 4)
    
    # Bellman-Ford theoretical: O(V*E)
    bf_theo = [n * e for n, e in zip(bf_filtered['nodes'], bf_filtered['edges'])]
    bf_scale = bf_filtered['execution_time'].iloc[-1] / bf_theo[-1] if bf_theo[-1] > 0 else 1
    bf_theo = [t * bf_scale for t in bf_theo]
    
    # Dijkstra theoretical: O((V+E)log(V))
    dj_theo = [(n + e) * np.log2(n) for n, e in zip(dj_filtered['nodes'], dj_filtered['edges'])]
    dj_scale = dj_filtered['execution_time'].iloc[-1] / dj_theo[-1] if dj_theo[-1] > 0 else 1
    dj_theo = [t * dj_scale for t in dj_theo]
    
    plt.plot(bf_filtered['nodes'], bf_filtered['execution_time'], 'o-', label='Bellman-Ford Actual', linewidth=2, markersize=8)
    plt.plot(bf_filtered['nodes'], bf_theo, 's--', label='Bellman-Ford Theory O(V*E)', linewidth=2, markersize=6)
    plt.plot(dj_filtered['nodes'], dj_filtered['execution_time'], 'o-', label='Dijkstra Actual', linewidth=2, markersize=8)
    plt.plot(dj_filtered['nodes'], dj_theo, 's--', label='Dijkstra Theory O((V+E)log(V))', linewidth=2, markersize=6)
    
    plt.title('Theoretical vs Actual Time Complexity')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("visualization", "algorithm_comparison.png"), dpi=300)
    print("Algorithm comparison visualization saved to visualization/algorithm_comparison.png")

def visualize_trust_network():
    """Create visualization of the Bitcoin Alpha trust network"""
    # Define file path
    data_file = "c:\\Users\\Hp\\Desktop\\Algo-Project\\soc-sign-bitcoinalpha.csv"
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Dataset file not found: {data_file}")
        return
    
    create_output_dir()
    
    # Load graph
    G = nx.DiGraph()
    trust_values = []
    
    print("Loading Bitcoin Alpha trust network...")
    with open(data_file, 'r') as file:
        # Skip comment line if present
        first_line = file.readline()
        if not first_line.startswith('//'):
            file.seek(0)
            
        for line in file:
            if not line.strip() or line.startswith('//'):
                continue
                
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
                
            source, target, rating, _ = parts
            source = int(source)
            target = int(target)
            rating = int(rating)
            
            G.add_edge(source, target, weight=rating)
            trust_values.append(rating)
    
    print(f"Network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Create trust distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(trust_values, bins=21, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Trust Ratings in Bitcoin Alpha Network')
    plt.xlabel('Trust Rating (-10 to +10)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("visualization", "trust_distribution.png"), dpi=300)
    print("Trust distribution visualization saved to visualization/trust_distribution.png")
    
    # Create network visualization (for a subset of the network)
    if G.number_of_nodes() > 500:
        print("Network is large. Visualizing a subset of the network...")
        # Get the most connected nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
        G_subset = G.subgraph(top_nodes)
    else:
        G_subset = G
    
    print(f"Visualizing network with {G_subset.number_of_nodes()} nodes and {G_subset.number_of_edges()} edges.")
    
    plt.figure(figsize=(12, 12))
    
    # Create layout
    pos = nx.spring_layout(G_subset, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_subset, pos, node_size=50, node_color='skyblue', alpha=0.8)
    
    # Create edge colors based on trust rating
    edge_colors = []
    edge_widths = []
    
    for u, v, data in G_subset.edges(data=True):
        weight = data.get('weight', 0)
        if weight > 0:
            edge_colors.append('green')
            edge_widths.append(0.5 + (weight / 10) * 2)  # Scale from 0.5 to 2.5
        else:
            edge_colors.append('red')
            edge_widths.append(0.5 + (abs(weight) / 10) * 2)  # Scale from 0.5 to 2.5
    
    # Draw edges
    nx.draw_networkx_edges(G_subset, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    
    plt.title('Bitcoin Alpha Trust Network Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("visualization", "bitcoin_alpha_trust_network.png"), dpi=300)
    print("Network visualization saved to visualization/bitcoin_alpha_trust_network.png")

def visualize_bellman_ford_iterations():
    """
    Create a visualization of Bellman-Ford algorithm iterations
    This function reads a trace file and visualizes how distances change across iterations
    """
    # Find the most recent trace file in the traces directory
    trace_files = [f for f in os.listdir("traces") if f.startswith("bellman_ford_trace_")]
    if not trace_files:
        print("No Bellman-Ford trace files found in the traces directory.")
        return
    
    # Use the latest trace file
    trace_file = os.path.join("traces", sorted(trace_files)[-1])
    
    create_output_dir()
    
    # Parse the trace file to extract iteration data
    iterations = []
    current_iteration = {}
    iteration_number = 0
    reading_distances = False
    
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Iteration"):
                iteration_number = int(line.split()[1].strip(':'))
                reading_distances = False
            
            elif line.startswith("Final distances from source node"):
                reading_distances = True
                
            elif reading_distances and ":" in line:
                parts = line.split(":")
                node = parts[0].strip().split()[1]
                distance_text = parts[1].strip()
                
                if distance_text == "Not reachable":
                    continue
                
                distance = float(distance_text)
                current_iteration[node] = distance
                
            elif line == "Algorithm completed. No negative cycles detected.":
                iterations.append(current_iteration.copy())
    
    # Check if we have data to visualize
    if not iterations:
        print("Could not extract iteration data from the trace file.")
        return
    
    # Get the source node and reachable nodes
    source_node = None
    all_nodes = set()
    
    for it in iterations:
        for node, distance in it.items():
            all_nodes.add(node)
            if distance == 0:
                source_node = node
    
    if not source_node:
        print("Could not identify source node in the trace file.")
        return
    
    print(f"Visualizing Bellman-Ford iterations from source node {source_node}.")
    
    # Create a plot showing how distances change across iterations
    plt.figure(figsize=(12, 8))
    
    for node in all_nodes:
        if node == source_node:
            continue  # Skip source node (always 0)
            
        distances = []
        for it in iterations:
            if node in it:
                distances.append(it[node])
            else:
                distances.append(float('inf'))
        
        # Only include nodes that have finite distances
        if any(d != float('inf') for d in distances):
            plt.plot(range(1, len(iterations) + 1), distances, 'o-', label=f"Node {node}")
    
    plt.title(f'Bellman-Ford Distance Convergence from Source Node {source_node}')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.grid(True)
    
    # Add legend if not too many nodes
    if len(all_nodes) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join("visualization", "bellman_ford_convergence.png"), dpi=300)
    print("Bellman-Ford convergence visualization saved to visualization/bellman_ford_convergence.png")

def main():
    """Main function to run visualizations"""
    print("Bellman-Ford Algorithm Visualization")
    print("====================================")
    
    while True:
        print("\nVisualization Menu:")
        print("1. Visualize Algorithm Performance Comparison")
        print("2. Visualize Bitcoin Alpha Trust Network")
        print("3. Visualize Bellman-Ford Iterations")
        print("4. Generate All Visualizations")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            visualize_algorithm_comparison()
        
        elif choice == '2':
            visualize_trust_network()
        
        elif choice == '3':
            visualize_bellman_ford_iterations()
        
        elif choice == '4':
            print("Generating all visualizations...")
            visualize_algorithm_comparison()
            visualize_trust_network()
            visualize_bellman_ford_iterations()
            print("All visualizations complete!")
        
        elif choice == '5':
            print("Exiting visualization tool...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
