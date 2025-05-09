"""
Prim's Algorithm Implementation for CS-2009 Design and Analysis of Algorithms Project
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
        # We treat the graph as undirected, so weight should be positive
        # We'll use the absolute value of the weight to ensure positivity
        weight = abs(float(weight))
        
        self.vertices.add(source)
        self.vertices.add(target)
        # Add edge in both directions since graph is undirected
        self.edges[source][target] = weight
        self.edges[target][source] = weight
    
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
                rating = float(rating)  # Convert rating to weight
                
                # Track unique nodes for limiting
                observed_nodes.add(source)
                observed_nodes.add(target)
                
                # Check if we've reached our node limit
                if limit and len(observed_nodes) > limit:
                    break
                
                self.add_edge(source, target, rating)
            
            print(f"Loaded graph with {len(self.vertices)} nodes and {sum(len(neighbors) for neighbors in self.edges.values())//2} edges")
            # Note: We divide edges by 2 since each edge is counted twice in undirected graph
    
    def prims_mst(self, start_vertex, trace_file):
        """
        Implement Prim's algorithm with efficient priority queue
        Returns the minimum spanning tree as a list of edges and total weight
        """
        # Initialize trace file
        with open(trace_file, 'w', encoding='utf-8') as f:
            f.write(f"Prim's Algorithm Trace from start vertex {start_vertex}\n")
            f.write(f"============================================\n\n")
        
        if start_vertex not in self.vertices:
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"Error: Start vertex {start_vertex} not found in the graph.\n")
            return [], 0
        
        # Priority queue entries are (weight, vertex, parent)
        pq = [(0, start_vertex, None)]
        
        # Keep track of visited vertices
        visited = set()
        
        # Store MST edges and total weight
        mst_edges = []
        total_weight = 0
        
        # Track step number for trace
        step = 1
        
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write("Initial state:\n")
            f.write(f"Start vertex: {start_vertex}\n")
            f.write(f"Priority Queue: [(0, {start_vertex}, None)]\n\n")
        
        while pq and len(visited) < len(self.vertices):
            weight, vertex, parent = heapq.heappop(pq)
            
            # Skip if vertex already visited
            if vertex in visited:
                continue
            
            # Add vertex to visited set
            visited.add(vertex)
            
            # Add edge to MST (except for start vertex)
            if parent is not None:
                mst_edges.append((parent, vertex, weight))
                total_weight += weight
            
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"Step {step}:\n")
                f.write(f"Selected vertex {vertex}" + (f" via edge ({parent}, {vertex}) with weight {weight}" if parent is not None else "") + "\n")
                f.write(f"Current MST weight: {total_weight}\n")
                f.write(f"Visited vertices: {visited}\n")
            
            # Add all edges from current vertex to priority queue
            for neighbor, edge_weight in self.edges[vertex].items():
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, neighbor, vertex))
                    with open(trace_file, 'a', encoding='utf-8') as f:
                        f.write(f"  Added edge ({vertex}, {neighbor}) with weight {edge_weight} to priority queue\n")
            
            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(f"  Current priority queue: {sorted(pq)}\n\n")
            
            step += 1
        
        # Write final results to trace file
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write("Algorithm completed.\n\n")
            f.write("Final Minimum Spanning Tree edges:\n")
            for edge in sorted(mst_edges):
                f.write(f"({edge[0]}, {edge[1]}) with weight {edge[2]}\n")
            f.write(f"\nTotal MST weight: {total_weight}\n")
        
        return mst_edges, total_weight

def save_results(mst_edges, total_weight, start_vertex, result_file):
    """Save the MST results to a file"""
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Minimum Spanning Tree from Start Vertex {start_vertex}\n")
        f.write(f"============================================\n\n")
        
        f.write("MST Edges:\n")
        for edge in sorted(mst_edges):
            f.write(f"({edge[0]}, {edge[1]}) with weight {edge[2]}\n")
        
        f.write(f"\nTotal MST weight: {total_weight}\n")

def run_performance_analysis(data_file):
    """Run performance analysis with different node sizes"""
    # Create output directories if they don't exist
    os.makedirs("performance", exist_ok=True)
    os.makedirs("performance/plots", exist_ok=True)
    
    # Node sizes for analysis
    node_sizes = [100, 200, 500, 1000, 2000, 3000]
    
    # Results will be stored here
    results = []
    
    # Start vertex for all tests (pick a vertex likely to exist in all subgraphs)
    start_vertex = None
    
    # Run Prim's for each node size
    for size in node_sizes:
        print(f"Running analysis for {size} nodes...")
        
        # Load graph with limited nodes
        graph = Graph()
        graph.load_from_file(data_file, size)
        
        # Find a suitable start vertex if not yet determined
        if start_vertex is None or start_vertex not in graph.vertices:
            start_vertex = next(iter(graph.vertices))
        
        # Set up file paths
        trace_file = os.path.join("performance", f"prims_trace_{size}_nodes.txt")
        
        # Measure execution time
        start_time = time.time()
        
        # Run Prim's algorithm
        mst_edges, total_weight = graph.prims_mst(start_vertex, trace_file)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Count edges in the graph (divide by 2 since graph is undirected)
        edge_count = sum(len(neighbors) for neighbors in graph.edges.values()) // 2
        
        # Calculate graph density
        max_possible_edges = len(graph.vertices) * (len(graph.vertices) - 1) // 2  # Divided by 2 for undirected graph
        density = edge_count / max_possible_edges if max_possible_edges > 0 else 0
        
        # Store results
        results.append({
            'nodes': len(graph.vertices),
            'edges': edge_count,
            'density': density,
            'mst_edges': len(mst_edges),
            'execution_time': execution_time
        })
        
        print(f"Completed analysis for {size} nodes:")
        print(f"  Actual node count: {len(graph.vertices)}")
        print(f"  Edge count: {edge_count}")
        print(f"  Density: {density:.6f}")
        print(f"  MST edges: {len(mst_edges)}")
        print(f"  Execution time: {execution_time:.6f} seconds")
    
    # Save results to CSV
    csv_file = os.path.join("performance", "prims_performance.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['nodes', 'edges', 'density', 'mst_edges', 'execution_time'])
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
    plt.title("Prim's Algorithm: Number of Nodes vs Execution Time")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "prims_nodes_vs_time.png"), dpi=300)
    
    # Plot 2: Edges vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(edges, times, 'o-', linewidth=2, markersize=8, color='green')
    plt.title("Prim's Algorithm: Number of Edges vs Execution Time")
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "prims_edges_vs_time.png"), dpi=300)
    
    # Plot 3: Graph Density vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(densities, times, 'o-', linewidth=2, markersize=8, color='red')
    plt.title("Prim's Algorithm: Graph Density vs Execution Time")
    plt.xlabel('Graph Density')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "prims_density_vs_time.png"), dpi=300)
    
    # Plot 4: Theoretical vs Actual Time Complexity
    # For Prim's with binary heap: O(E log V)
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical complexity: O(E log V)
    theoretical = [e * np.log2(n) for n, e in zip(nodes, edges)]
    
    # Scale to match actual times
    scale_factor = times[-1] / theoretical[-1] if theoretical[-1] > 0 else 1
    theoretical = [t * scale_factor for t in theoretical]
    
    plt.plot(nodes, times, 'o-', linewidth=2, markersize=8, label='Actual Time')
    plt.plot(nodes, theoretical, 's--', linewidth=2, markersize=8, label='Theoretical O(E log V)')
    
    plt.title("Prim's Algorithm: Theoretical vs Actual Time Complexity")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("performance", "plots", "prims_theoretical_vs_actual.png"), dpi=300)
    
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
        print("\nPrim's Algorithm Menu:")
        print("1. Run for the whole dataset")
        print("2. Run performance analysis")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Get start vertex from user
            while True:
                try:
                    start_vertex = int(input("Enter start vertex ID: "))
                    if start_vertex in graph.vertices:
                        break
                    else:
                        print(f"Vertex {start_vertex} not found in the graph. Please try again.")
                except ValueError:
                    print("Please enter a valid integer.")
            
            # Set up file paths
            result_file = os.path.join("results", f"prims_results_{start_vertex}.txt")
            trace_file = os.path.join("traces", f"prims_trace_{start_vertex}.txt")
            time_file = os.path.join("times", f"prims_time_{start_vertex}.txt")
            
            print(f"Running Prim's algorithm from start vertex {start_vertex}...")
            
            # Measure execution time
            start_time = time.time()
            
            # Run Prim's algorithm
            mst_edges, total_weight = graph.prims_mst(start_vertex, trace_file)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            save_results(mst_edges, total_weight, start_vertex, result_file)
            
            # Save execution time
            with open(time_file, 'w') as f:
                f.write(f"Prim's Algorithm Execution Time\n")
                f.write(f"Start vertex: {start_vertex}\n")
                f.write(f"Number of nodes: {len(graph.vertices)}\n")
                f.write(f"Number of edges: {sum(len(neighbors) for neighbors in graph.edges.values())//2}\n")
                f.write(f"MST edges: {len(mst_edges)}\n")
                f.write(f"Total MST weight: {total_weight}\n")
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
