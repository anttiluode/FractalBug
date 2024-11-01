# Fractal Adaptive System


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
git clone https://github.com/anttiluode/fractalbug.git

cd fractalbug

Install Dependencies

pip install -r requirements.txt

Run the application using:

python app.py
```

Controls
Start: Begins the adaptive system's processing loop.
Stop: Halts the processing loop.
Visualize Nodes: Opens a separate window displaying a 3D visualization of the network's nodes and connections.
Config: Opens the configuration window to adjust system parameters.
Save: Saves the current system configuration and node states to a JSON file.
Load: Loads a system configuration and node states from a JSON file.
Key Differences from Previous Version (fractal bug)
The Fractal Adaptive System represents a significant enhancement over the earlier version, referred to as the fractal bug. Below are the key differences:

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

This project is licensed under the MIT License.
