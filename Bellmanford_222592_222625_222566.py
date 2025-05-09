"""
Bellman-Ford Algorithm Implementation for CS-2009 Design and Analysis of Algorithms Project
Group: 222592, 222625, 222566
"""
import os
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = []
        self.adj_list = defaultdict(dict)
    
    def add_edge(self, source, target, weight):
        # Convert ratings from -10 to +10 range to positive weights for Bellman-Ford
        # We invert the rating since higher trust (-10 to +10) means shorter path
        # Add 11 to ensure all weights are positive
        adjusted_weight = 21 - (weight + 10)  # Range: 1 (high trust) to 21 (low trust)
        
        self.vertices.add(source)
        self.vertices.add(target)
        self.edges.append((source, target, adjusted_weight))
        self.adj_list[source][target] = adjusted_weight
    
    def load_from_file(self, file_path, limit=None):
        """Load graph from CSV file with optional limit on number of nodes"""
        with open(file_path, 'r') as file:
            # Skip comment line if present
            first_line = file.readline()
            if first_line.startswith('//'):
                pass  # Skip comment line
            else:
                # Reset file pointer if first line was not a comment
                file.seek(0)
            
            node_count = 0
            observed_nodes = set()
            
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
                
                self.add_edge(source, target, rating)
            
            print(f"Loaded graph with {len(self.vertices)} nodes and {len(self.edges)} edges")
    
    def bellman_ford(self, source, trace_file):
        """
        Implement Bellman-Ford algorithm
        Traces the complete algorithm execution process
        Returns distances and predecessors or None if negative cycle detected
        """
        # Initialize trace file
        with open(trace_file, 'w', encoding='utf-8') as f:
            f.write(f"Bellman-Ford Algorithm Trace from source node {source}\n")
            f.write(f"================================================\n\n")
        
        if source not in self.vertices:
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"Error: Source node {source} not found in the graph.\n")
            return {}, []
        
        # Initialize distances with infinity for all nodes except source
        distances = {node: float('infinity') for node in self.vertices}
        distances[source] = 0
        
        # Initialize predecessors for path reconstruction
        predecessors = {node: None for node in self.vertices}
        
        # Trace initial state
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write("Initial state:\n")
            f.write(f"Distance to source node {source}: 0\n")
            f.write(f"All other distances set to infinity\n\n")
        
        # Main Bellman-Ford algorithm
        # Relax edges |V| - 1 times
        iteration = 1
        
        for _ in range(len(self.vertices) - 1):
            # Track if any changes were made in this iteration
            changes_made = False
            
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"Iteration {iteration}:\n")
            
            # For each edge in the graph
            for u, v, weight in self.edges:
                if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                    # Relaxation step
                    old_distance = distances[v]
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    changes_made = True
                    
                    with open(trace_file, 'a', encoding='utf-8') as f:
                        f.write(f"  Relaxed edge ({u}, {v}) with weight {weight}:\n")
                        f.write(f"    Updated distance to node {v}: {old_distance} -> {distances[v]}\n")
                        f.write(f"    Updated predecessor of {v} to {u}\n")
            
            # If no changes were made in this iteration, we can terminate early
            if not changes_made:
                with open(trace_file, 'a', encoding='utf-8') as f:
                    f.write(f"  No changes made in this iteration. Early termination.\n\n")
                break
            else:
                with open(trace_file, 'a', encoding='utf-8') as f:
                    f.write("\n")
            
            iteration += 1
        
        # Check for negative cycles
        for u, v, weight in self.edges:
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                with open(trace_file, 'a', encoding='utf-8') as f:
                    f.write("Negative cycle detected! The graph contains a negative weight cycle.\n")
                    f.write(f"Evidence: Edge ({u}, {v}) with weight {weight} can still be relaxed.\n")
                    f.write(f"Current distance to {v}: {distances[v]}, but path through {u} gives: {distances[u] + weight}\n")
                return None, None
        
        # Final distances
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write(f"Algorithm completed. No negative cycles detected.\n\n")
            f.write(f"Final distances from source node {source}:\n")
            
            for node, distance in sorted(distances.items()):
                if distance == float('infinity'):
                    f.write(f"Node {node}: Not reachable\n")
                else:
                    f.write(f"Node {node}: {distance}\n")
        
        return distances, predecessors
    
    def reconstruct_path(self, source, target, predecessors):
        """Reconstruct the shortest path from source to target"""
        if target not in predecessors or (predecessors[target] is None and source != target):
            return []  # No path exists
        
        path = []
        current = target
        
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        # Reverse to get path from source to target
        path.reverse()
        return path

def save_results(distances, predecessors, source, result_file):
    """Save the shortest paths to a file"""
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Bellman-Ford: Shortest Paths from Source Node {source}\n")
        f.write(f"===============================================\n\n")
        
        if distances is None:
            f.write("The graph contains a negative weight cycle.\n")
            f.write("Cannot compute shortest paths.\n")
            return
        
        for target in sorted(distances.keys()):
            distance = distances[target]
            
            if distance == float('infinity'):
                f.write(f"Node {target}: Not reachable\n")
            else:
                # Reconstruct path
                path = []
                current = target
                
                while current is not None:
                    path.append(current)
                    current = predecessors.get(current)
                
                # Reverse to get path from source to target
                path.reverse()
                
                f.write(f"Node {target}: Distance = {distance}, Path = {' -> '.join(map(str, path))}\n")

def run_performance_analysis(data_file):
    """Run performance analysis with different node sizes"""
    # Create output directories if they don't exist
    os.makedirs("performance", exist_ok=True)
    os.makedirs("performance/plots", exist_ok=True)
    
    # Node sizes for analysis
    node_sizes = [100, 200, 500, 1000, 2000]
    
    # Results will be stored here
    results = []
    
    # Source node for all tests (pick a node likely to exist in all subgraphs)
    source_node = None
    
    # Run Bellman-Ford for each node size
    for size in node_sizes:
        print(f"Running analysis for {size} nodes...")
        
        # Load graph with limited nodes
        graph = Graph()
        graph.load_from_file(data_file, size)
        
        # Find a suitable source node if not yet determined
        if source_node is None or source_node not in graph.vertices:
            source_node = next(iter(graph.vertices))
        
        # Set up file paths
        trace_file = os.path.join("performance", f"bellman_ford_trace_{size}_nodes.txt")
        
        # Measure execution time
        start_time = time.time()
        
        # Run Bellman-Ford algorithm
        distances, _ = graph.bellman_ford(source_node, trace_file)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Count number of reachable nodes (non-infinite distances)
        reachable_nodes = 0
        if distances:  # Check if distances is not None (no negative cycle)
            reachable_nodes = sum(1 for d in distances.values() if d != float('infinity'))
        
        # Count edges in the graph
        edge_count = len(graph.edges)
        
        # Calculate graph density
        max_possible_edges = len(graph.vertices) * (len(graph.vertices) - 1)
        density = edge_count / max_possible_edges if max_possible_edges > 0 else 0
        
        # Store results
        results.append({
            'nodes': len(graph.vertices),
            'edges': edge_count,
            'density': density,
            'reachable_nodes': reachable_nodes,
            'execution_time': execution_time
        })
        
        print(f"Completed analysis for {size} nodes:")
        print(f"  Actual node count: {len(graph.vertices)}")
        print(f"  Edge count: {edge_count}")
        print(f"  Density: {density:.6f}")
        print(f"  Execution time: {execution_time:.6f} seconds")
    
    # Save results to CSV
    csv_file = os.path.join("performance", "bellman_ford_performance.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['nodes', 'edges', 'density', 'reachable_nodes', 'execution_time'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Performance results saved to {csv_file}")
    
    # Generate plots
    generate_plots(results)

def generate_plots(results):
    """Generate performance analysis plots"""
    # Extract data for plotting
    nodes = [r['nodes'] for r in results]
    times = [r['execution_time'] for r in results]
    edges = [r['edges'] for r in results]
    densities = [r['density'] for r in results]
    
    # Plot 1: Nodes vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, times, 'o-', linewidth=2, markersize=8)
    plt.title('Bellman-Ford Algorithm: Number of Nodes vs Execution Time')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "bellman_ford_nodes_vs_time.png"), dpi=300)
    
    # Plot 2: Edges vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(edges, times, 'o-', linewidth=2, markersize=8, color='green')
    plt.title('Bellman-Ford Algorithm: Number of Edges vs Execution Time')
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "bellman_ford_edges_vs_time.png"), dpi=300)
    
    # Plot 3: Graph Density vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(densities, times, 'o-', linewidth=2, markersize=8, color='red')
    plt.title('Bellman-Ford Algorithm: Graph Density vs Execution Time')
    plt.xlabel('Graph Density')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "bellman_ford_density_vs_time.png"), dpi=300)
      # Plot 4: Theoretical vs Actual Time Complexity
    # For Bellman-Ford: O(V*E)
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical complexity: O(V*E)
    theoretical = [n * e for n, e in zip(nodes, edges)]
    
    # Scale to match actual times
    scale_factor = times[-1] / theoretical[-1] if theoretical[-1] > 0 else 1
    theoretical = [t * scale_factor for t in theoretical]
    
    plt.plot(nodes, times, 'o-', linewidth=2, markersize=8, label='Actual Time')
    plt.plot(nodes, theoretical, 's--', linewidth=2, markersize=8, label='Theoretical O(V*E)')
    
    plt.title('Bellman-Ford Algorithm: Theoretical vs Actual Time Complexity')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "bellman_ford_theoretical_vs_actual.png"), dpi=300)
    
    print("Performance plots generated in the 'performance/plots' directory")

def main():
    # Create output directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("traces", exist_ok=True)
    os.makedirs("times", exist_ok=True)
    
    graph = Graph()
    
    # Get the current directory
    data_file = "c:\\Users\\Hp\\Desktop\\Algo-Project\\soc-sign-bitcoinalpha.csv"
    
    print("Loading Bitcoin Alpha dataset...")
    graph.load_from_file(data_file)
    
    while True:
        print("\nBellman-Ford Algorithm Menu:")
        print("1. Run for the whole dataset")
        print("2. Run performance analysis")
        print("3. Compare with Dijkstra algorithm")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            # Get source node from user
            while True:
                try:
                    source_node = int(input("Enter source node ID: "))
                    if source_node in graph.vertices:
                        break
                    else:
                        print(f"Node {source_node} not found in the graph. Please try again.")
                except ValueError:
                    print("Please enter a valid integer.")
            
            # Set up file paths
            result_file = os.path.join("results", f"bellman_ford_results_{source_node}.txt")
            trace_file = os.path.join("traces", f"bellman_ford_trace_{source_node}.txt")
            time_file = os.path.join("times", f"bellman_ford_time_{source_node}.txt")
            
            print(f"Running Bellman-Ford algorithm from source node {source_node}...")
            
            # Measure execution time
            start_time = time.time()
            
            # Run Bellman-Ford algorithm
            distances, predecessors = graph.bellman_ford(source_node, trace_file)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            save_results(distances, predecessors, source_node, result_file)
            
            # Save execution time
            with open(time_file, 'w') as f:
                f.write(f"Bellman-Ford Algorithm Execution Time\n")
                f.write(f"Source node: {source_node}\n")
                f.write(f"Number of nodes: {len(graph.vertices)}\n")
                f.write(f"Number of edges: {len(graph.edges)}\n")
                f.write(f"Execution time: {execution_time:.6f} seconds\n")
                
                if distances is None:
                    f.write("Note: Negative cycle detected in the graph.\n")
            
            print(f"Results saved to {result_file}")
            print(f"Trace saved to {trace_file}")
            print(f"Execution time saved to {time_file}")
            print(f"Execution time: {execution_time:.6f} seconds")
            
            if distances is None:
                print("Note: Negative cycle detected in the graph.")
        
        elif choice == '2':
            print("Running performance analysis...")
            run_performance_analysis(data_file)
        
        elif choice == '3':
            print("Comparing Bellman-Ford with Dijkstra algorithm...")
            compare_algorithms(data_file)
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

def compare_algorithms(data_file):
    """Compare Bellman-Ford with Dijkstra algorithm"""
    # Create output directories if they don't exist
    os.makedirs("comparison", exist_ok=True)
    
    # Node sizes for comparison
    node_sizes = [100, 500, 1000]
    
    # Results will be stored here
    results = []
    
    # Source node for all tests
    source_node = None
    
    for size in node_sizes:
        print(f"Comparing algorithms for {size} nodes...")
        
        # Load graph for Bellman-Ford
        bf_graph = Graph()
        bf_graph.load_from_file(data_file, size)
        
        # Find a suitable source node if not yet determined
        if source_node is None or source_node not in bf_graph.vertices:
            source_node = next(iter(bf_graph.vertices))
        
        # Set up file paths
        bf_trace_file = os.path.join("comparison", f"bellman_ford_trace_{size}_nodes.txt")
        
        # Run Bellman-Ford and measure time
        bf_start_time = time.time()
        bf_distances, _ = bf_graph.bellman_ford(source_node, bf_trace_file)
        bf_time = time.time() - bf_start_time
        
        # Load graph for Dijkstra (need to create a compatible Graph class instance)
        # Here we're importing the Dijkstra Graph class to use its implementation
        from Dijkstra_222592_222625_222566 import Graph as DijkstraGraph
        
        dijkstra_graph = DijkstraGraph()
        dijkstra_graph.load_from_file(data_file, size)
        
        # Set up file paths for Dijkstra
        dijkstra_trace_file = os.path.join("comparison", f"dijkstra_trace_{size}_nodes.txt")
        
        # Run Dijkstra and measure time
        dijkstra_start_time = time.time()
        dijkstra_distances, _ = dijkstra_graph.dijkstra(source_node, dijkstra_trace_file)
        dijkstra_time = time.time() - dijkstra_start_time
        
        # Store results
        results.append({
            'nodes': size,
            'bellman_ford_time': bf_time,
            'dijkstra_time': dijkstra_time,
            'time_ratio': bf_time / dijkstra_time if dijkstra_time > 0 else float('inf')
        })
        
        print(f"  Bellman-Ford time: {bf_time:.6f} seconds")
        print(f"  Dijkstra time: {dijkstra_time:.6f} seconds")
        print(f"  Bellman-Ford is {bf_time / dijkstra_time:.2f}x slower than Dijkstra")
    
    # Save results to CSV
    csv_file = os.path.join("comparison", "algorithm_comparison.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['nodes', 'bellman_ford_time', 'dijkstra_time', 'time_ratio'])
        writer.writeheader()
        writer.writerows(results)
    
    # Generate comparison plot
    plt.figure(figsize=(12, 6))
    
    nodes = [r['nodes'] for r in results]
    bf_times = [r['bellman_ford_time'] for r in results]
    dijkstra_times = [r['dijkstra_time'] for r in results]
    
    bar_width = 0.35
    index = np.arange(len(nodes))
    
    plt.bar(index, bf_times, bar_width, label='Bellman-Ford')
    plt.bar(index + bar_width, dijkstra_times, bar_width, label='Dijkstra')
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Bellman-Ford vs Dijkstra: Performance Comparison')
    plt.xticks(index + bar_width / 2, nodes)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join("comparison", "algorithm_comparison.png"), dpi=300)
    
    print(f"Comparison results saved to {csv_file}")
    print(f"Comparison plot saved to comparison/algorithm_comparison.png")

if __name__ == "__main__":
    main()
