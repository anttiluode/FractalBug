# Fractal Adaptive System

![Fractal Adaptive System](https://example.com/fractal-adaptive-system-banner.png)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Key Differences from Previous Version](#key-differences-from-previous-version)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Saving and Loading System State](#saving-and-loading-system-state)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **Fractal Adaptive System** is an advanced AI simulation that leverages fractal geometry principles to create a highly scalable and modular neural network architecture. Inspired by recent research on sparse autoencoders and their geometric feature structures, this system introduces recursive layering and sophisticated visualization techniques to emulate complex, self-similar patterns found in natural and artificial intelligence systems.

## Features

- **Recursive Fractal Architecture:** 
  - Each layer (`AdaptiveNetwork`) can contain multiple sub-layers, creating a self-similar, hierarchical structure.
  - Dynamic addition and removal of sub-layers based on growth rates and pruning thresholds.

- **Advanced Visualization:**
  - **2D Fractal Visualization:** Recursive rendering of nodes and connections using Tkinter.
  - **3D Visualization with Matplotlib:** Immersive 3D plots of the network's fractal structure.
  - **Real-Time Node Visualization:** Separate window displaying node positions in 3D space.

- **Dynamic Adaptation:**
  - Hebbian learning rules for adaptive connections.
  - Probabilistic growth and pruning of sub-layers to maintain optimal network complexity.

- **Comprehensive Configuration:**
  - Adjustable parameters including depth, growth rate, pruning threshold, and more.
  - Webcam selection for sensory input processing.

- **State Management:**
  - Save and load system configurations and node states via JSON files.

- **Robust Logging:**
  - Detailed logging of system events and errors for easier debugging and monitoring.

## Installation

### Prerequisites

- **Python 3.7 or higher**  
Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/fractal-adaptive-system.git
cd fractal-adaptive-system
Install Dependencies
It's recommended to use a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Usage
Run the application using:

bash
Copy code
python fractal_adaptive_system.py
Controls
Start: Begins the adaptive system's processing loop.
Stop: Halts the processing loop.
Visualize Nodes: Opens a separate window displaying a 3D visualization of the network's nodes and connections.
Config: Opens the configuration window to adjust system parameters.
Save: Saves the current system configuration and node states to a JSON file.
Load: Loads a system configuration and node states from a JSON file.
Key Differences from Previous Version (fractal bug)
The Fractal Adaptive System represents a significant enhancement over the earlier version, referred to as the fractal bug. Below are the key differences:

Feature	Fractal Bug (Earlier Version)	Fractal Adaptive System (Current Version)
Architecture	Flat multi-layered structure without recursion.	Recursive, hierarchical, fractal-inspired architecture with nested sub-layers.
Layer Management	Linear addition/removal of layers.	Dynamic, recursive sub-layer management allowing self-similarity.
Connection Patterns	Simple, potentially random connections between nodes.	Fractal-inspired connection patterns, such as connecting nodes based on powers of two differences.
Visualization	Basic 2D Tkinter canvas visualization.	Enhanced visualization including recursive 2D rendering and advanced 3D Matplotlib visualizations.
Configuration	Basic parameters (e.g., node counts, growth rates).	Expanded configuration with fractal-specific settings like depth, alongside existing parameters.
Adaptive Mechanisms	Basic activation and Hebbian adaptation.	Hierarchical, recursive processing and adaptive Hebbian learning across all fractal layers.
Clustering & Dimensionality Reduction	Basic clustering methods without advanced dimensionality reduction.	Integration of advanced techniques like Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) to reveal fractal patterns.
Logging & Error Handling	Basic logging for system events and errors.	Detailed, fractal-aware logging and robust error management for recursive processes.
User Interface	Basic controls and single visualization window.	Enhanced UI with multiple visualization tools, advanced configuration options, and interactive elements.
Theoretical Alignment	Foundational geometric concepts.	Direct alignment with fractal geometry insights from recent AI research, inspired by MIT's study on sparse autoencoders.
Summary: The current version introduces a recursive fractal architecture, significantly enhancing the system's complexity, scalability, and visual interpretability. Advanced visualization techniques, dynamic sub-layer management, and alignment with fractal geometry principles set it apart from the earlier fractal bug version, making it a more robust and insightful tool for exploring adaptive neural networks.

Visualization
2D Fractal Visualization
The FractalNodeVisualizer class provides a Tkinter-based canvas that recursively draws nodes and their connections, reflecting the hierarchical fractal structure of the network.

3D Visualization with Matplotlib
The FractalMatplotlibVisualizer class utilizes Matplotlib's 3D plotting capabilities to render an immersive, three-dimensional view of the network's nodes and connections.

Real-Time Node Visualization
The NodeVisualizer class opens a separate window displaying a real-time scatter plot of node positions in 3D space, offering an additional perspective on the network's structure.

Configuration
Access the configuration window by clicking the Config button. Adjust the following parameters:

Depth: Controls the recursion level of the network's fractal architecture.
Pruning Threshold: Determines the threshold for removing sub-layers.
Growth Rate: Sets the probability of adding a new sub-layer at each growth interval.
Minimum Nodes: Specifies the minimum number of nodes per layer.
Maximum Nodes: Defines the maximum number of nodes per layer.
Webcam Selection: Choose the desired webcam for sensory input processing.
Note: After making changes, click Apply to update the system's configuration.

Saving and Loading System State
Use the Save and Load buttons to persist and retrieve the system's configuration and node states.

Save: Serializes the current system state to a JSON file.
Load: Deserializes the system state from a JSON file, updating configurations and node connections accordingly.
Logging
All system events, including initialization, node and sub-layer management, visualization updates, and errors, are logged to both the console and a system.log file for monitoring and debugging purposes.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
License
This project is licensed under the MIT License.