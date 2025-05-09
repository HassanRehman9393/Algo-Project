"""
Dijkstra's Algorithm Implementation for CS-2009 Design and Analysis of Algorithms Project
Group: 222592, 222625, 222566
"""
import os
import heapq
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = defaultdict(dict)
    
    def add_edge(self, source, target, weight):
        # Convert ratings from -10 to +10 range to positive weights for Dijkstra
        # We invert the rating since higher trust (-10 to +10) means shorter path
        # Add 11 to ensure all weights are positive
        adjusted_weight = 21 - (weight + 10)  # Range: 1 (high trust) to 21 (low trust)
        
        self.vertices.add(source)
        self.vertices.add(target)
        self.edges[source][target] = adjusted_weight
    
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
            
            print(f"Loaded graph with {len(self.vertices)} nodes and {sum(len(neighbors) for neighbors in self.edges.values())} edges")
    
    def dijkstra(self, source, trace_file):
        """
        Implement Dijkstra's algorithm with efficient priority queue
        Traces the complete algorithm execution process
        """
        # Initialize trace file
        with open(trace_file, 'w', encoding='utf-8') as f:
            f.write(f"Dijkstra's Algorithm Trace from source node {source}\n")
            f.write(f"============================================\n\n")
        
        if source not in self.vertices:
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"Error: Source node {source} not found in the graph.\n")
            return {}, []
        
        # Initialize distances with infinity for all nodes except source
        distances = {node: float('infinity') for node in self.vertices}
        distances[source] = 0
        
        # Initialize predecessors for path reconstruction
        predecessors = {node: None for node in self.vertices}
        
        # Priority queue for efficient node selection
        # Format: (distance, node)
        priority_queue = [(0, source)]
        
        # Visited nodes
        visited = set()
        
        # Trace steps
        step = 1
        
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write("Initial state:\n")
            f.write(f"Distance to source node {source}: 0\n")
            f.write(f"All other distances set to infinity\n\n")
            f.write(f"Priority Queue: [(0, {source})]\n\n")
        
        while priority_queue:
            # Get node with minimum distance
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # Skip if we've already processed this node
            if current_node in visited:
                continue
            
            # Mark as visited
            visited.add(current_node)
            
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"Step {step}:\n")
                f.write(f"Removed node {current_node} from priority queue with distance {current_distance}\n")
                f.write(f"Visited nodes: {visited}\n")
            
            # Process neighbors
            for neighbor, weight in self.edges.get(current_node, {}).items():
                if neighbor in visited:
                    with open(trace_file, 'a', encoding='utf-8') as f:
                        f.write(f"  Neighbor {neighbor} already visited, skipping\n")
                    continue
                
                # Calculate new distance
                new_distance = distances[current_node] + weight
                
                # Update if we found a shorter path
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    
                    # Add to priority queue
                    heapq.heappush(priority_queue, (new_distance, neighbor))
                    
                    with open(trace_file, 'a', encoding='utf-8') as f:
                        f.write(f"  Updated distance to node {neighbor}: {new_distance}\n")
                        f.write(f"  Updated predecessor of {neighbor} to {current_node}\n")
                        f.write(f"  Added ({new_distance}, {neighbor}) to priority queue\n")
                else:
                    with open(trace_file, 'a', encoding='utf-8') as f:
                        f.write(f"  No update for node {neighbor}, current distance {distances[neighbor]} is better than new distance {new_distance}\n")
            
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"  Current priority queue: {sorted(priority_queue)}\n\n")
            
            step += 1
        
        # Final distances
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write(f"Algorithm completed.\n\n")
            f.write(f"Final distances from source node {source}:\n")
            
            for node, distance in sorted(distances.items()):
                if distance == float('infinity'):
                    f.write(f"Node {node}: Not reachable\n")
                else:
                    f.write(f"Node {node}: {distance}\n")
        
        return distances, predecessors
    
    def reconstruct_path(self, source, target, predecessors):
        """Reconstruct the shortest path from source to target"""
        if target not in predecessors or predecessors[target] is None and source != target:
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
        f.write(f"Shortest Paths from Source Node {source}\n")
        f.write(f"===================================\n\n")
        
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
    node_sizes = [100, 200, 500, 1000, 2000, 3000]
    
    # Results will be stored here
    results = []
    
    # Source node for all tests (pick a node likely to exist in all subgraphs)
    source_node = None
    
    # Run Dijkstra for each node size
    for size in node_sizes:
        print(f"Running analysis for {size} nodes...")
        
        # Load graph with limited nodes
        graph = Graph()
        graph.load_from_file(data_file, size)
        
        # Find a suitable source node if not yet determined
        if source_node is None or source_node not in graph.vertices:
            source_node = next(iter(graph.vertices))
        
        # Set up file paths
        trace_file = os.path.join("performance", f"dijkstra_trace_{size}_nodes.txt")
        
        # Measure execution time
        start_time = time.time()
        
        # Run Dijkstra's algorithm
        distances, _ = graph.dijkstra(source_node, trace_file)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Count number of reachable nodes (non-infinite distances)
        reachable_nodes = sum(1 for d in distances.values() if d != float('infinity'))
        
        # Count edges in the graph
        edge_count = sum(len(neighbors) for neighbors in graph.edges.values())
        
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
    csv_file = os.path.join("performance", "dijkstra_performance.csv")
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
    plt.title('Dijkstra Algorithm: Number of Nodes vs Execution Time')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "nodes_vs_time.png"), dpi=300)
    
    # Plot 2: Edges vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(edges, times, 'o-', linewidth=2, markersize=8, color='green')
    plt.title('Dijkstra Algorithm: Number of Edges vs Execution Time')
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "edges_vs_time.png"), dpi=300)
    
    # Plot 3: Graph Density vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(densities, times, 'o-', linewidth=2, markersize=8, color='red')
    plt.title('Dijkstra Algorithm: Graph Density vs Execution Time')
    plt.xlabel('Graph Density')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "density_vs_time.png"), dpi=300)
    
    # Plot 4: Theoretical vs Actual Time Complexity
    # For Dijkstra with binary heap: O((V+E)logV)
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical complexity: O((V+E)logV)
    theoretical = [(n + e) * np.log2(n) for n, e in zip(nodes, edges)]
    
    # Scale to match actual times
    scale_factor = times[-1] / theoretical[-1] if theoretical[-1] > 0 else 1
    theoretical = [t * scale_factor for t in theoretical]
    
    plt.plot(nodes, times, 'o-', linewidth=2, markersize=8, label='Actual Time')
    plt.plot(nodes, theoretical, 's--', linewidth=2, markersize=8, label='Theoretical O((V+E)logV)')
    
    plt.title('Dijkstra Algorithm: Theoretical vs Actual Time Complexity')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "theoretical_vs_actual.png"), dpi=300)
    
    print("Performance plots generated in the 'performance/plots' directory")

def main():
    # Create output directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("traces", exist_ok=True)
    os.makedirs("times", exist_ok=True)
    
    graph = Graph()
      # Get the current directory
    data_file = os.path.join(os.path.dirname(__file__), "soc-sign-bitcoinalpha.csv")
    
    print("Loading Bitcoin Alpha dataset...")
    graph.load_from_file(data_file)
    
    while True:
        print("\nDijkstra's Algorithm Menu:")
        print("1. Run for the whole dataset")
        print("2. Run performance analysis")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
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
            result_file = os.path.join("results", f"dijkstra_results_{source_node}.txt")
            trace_file = os.path.join("traces", f"dijkstra_trace_{source_node}.txt")
            time_file = os.path.join("times", f"dijkstra_time_{source_node}.txt")
            
            print(f"Running Dijkstra's algorithm from source node {source_node}...")
            
            # Measure execution time
            start_time = time.time()
            
            # Run Dijkstra's algorithm
            distances, predecessors = graph.dijkstra(source_node, trace_file)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            save_results(distances, predecessors, source_node, result_file)
            
            # Save execution time
            with open(time_file, 'w') as f:
                f.write(f"Dijkstra's Algorithm Execution Time\n")
                f.write(f"Source node: {source_node}\n")
                f.write(f"Number of nodes: {len(graph.vertices)}\n")
                f.write(f"Number of edges: {sum(len(neighbors) for neighbors in graph.edges.values())}\n")
                f.write(f"Execution time: {execution_time:.6f} seconds\n")
            
            print(f"Results saved to {result_file}")
            print(f"Trace saved to {trace_file}")
            print(f"Execution time saved to {time_file}")
            print(f"Execution time: {execution_time:.6f} seconds")
        
        elif choice == '2':
            print("Running performance analysis...")
            # Run the performance analysis function directly
            run_performance_analysis(data_file)
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
