# Dijkstra's Algorithm Implementation for Bitcoin Alpha Trust Network

## Project Overview
This project implements Dijkstra's algorithm to find the shortest path in the Bitcoin Alpha trust network. The implementation includes complete algorithm tracing, performance analysis, and visualization of the network.

## Dataset
The implementation uses the Bitcoin Alpha trust network dataset, which is a who-trusts-whom network of people who trade using Bitcoin on a platform called Bitcoin Alpha. Each edge represents a trust rating between users ranging from -10 (total distrust) to +10 (total trust).

Dataset statistics:
- Nodes: 3,783
- Edges: 24,186
- Range of edge weights: -10 to +10
- Percentage of positive edges: 93%

## Files Structure
- `run.py`: Main entry point to run the project
- `Dijkstra_main.py`: Main implementation of Dijkstra's algorithm
- `Dijkstra_performance.py`: Performance analysis module
- `Dijkstra_visualization.py`: Network visualization module
- `soc-sign-bitcoinalpha.csv`: The dataset file

## Requirements
- Python 3.6 or higher
- Required Python packages:
  - networkx
  - matplotlib
  - numpy
  - pandas

## How to Run the Project

1. Make sure you have Python installed on your system.

2. Run the main script:
   ```
   python run.py
   ```

3. Follow the menu options:
   - Option 1: Run Dijkstra's Algorithm
     - Select to run on the whole dataset or perform performance analysis
     - Enter a source node when prompted
   - Option 2: Visualize the Bitcoin Alpha Network
   - Option 3: Install Required Dependencies
   - Option 4: Exit

## Implementation Details

### Dijkstra's Algorithm
- Implements Dijkstra's algorithm using a priority queue (min-heap) for efficient node selection
- Handles the weighted directed graph structure of the Bitcoin Alpha trust network
- Converts trust ratings to appropriate path weights for shortest path calculation
- Provides complete tracing of the algorithm execution process

### Performance Analysis
- Analyzes algorithm performance with different node sizes: 100, 200, 500, 1000, 2000, 3000
- Generates performance plots:
  - Number of Nodes vs Execution Time
  - Number of Edges vs Execution Time
  - Graph Density vs Execution Time
  - Theoretical vs Actual Time Complexity

### Visualization
- Visualizes the Bitcoin Alpha trust network as a directed graph
- Edge colors represent trust levels (red for negative, green for positive)
- Node sizes reflect user importance in the network
- Provides a trust distribution histogram

## Output Files
- `results/`: Contains shortest path results for each run
- `traces/`: Contains detailed algorithm execution traces
- `times/`: Contains execution time measurements
- `performance/`: Contains performance analysis results and plots
- `visualization/`: Contains network visualizations

## Time Complexity Analysis
- Dijkstra's algorithm with binary heap: O((V+E)log V)
  - V: Number of vertices (users in the network)
  - E: Number of edges (trust ratings between users)
- The implementation optimizes for efficiency with priority queue-based node selection

## Note on Graph Weights
The trust ratings from -10 to +10 are converted to appropriate path weights for Dijkstra's algorithm, where:
- Higher trust ratings correspond to shorter path lengths
- Lower trust ratings correspond to longer path lengths

This conversion ensures that paths through highly trusted users are preferred.
