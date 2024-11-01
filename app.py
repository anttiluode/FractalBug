import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import threading
import queue
import logging
import time
from collections import deque
from typing import Dict, Any, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)

class SystemConfig:
    def __init__(self):
        self.display_width = 800
        self.display_height = 600
        self.initial_nodes = 20
        self.initial_sub_layers = 2
        self.min_nodes = 50
        self.max_nodes = 500
        self.growth_rate = 0.1  # 10% chance to add a node every 100ms
        self.pruning_threshold = 0.3
        self.camera_index = 0  # Default camera index
        self.vision_cone_length = 150
        self.movement_speed = 3.0
        self.max_sub_layers = 4
        self.depth = 1  # Placeholder for depth parameter

    def to_dict(self):
        """Serialize the configuration to a dictionary."""
        return {
            'display_width': self.display_width,
            'display_height': self.display_height,
            'initial_nodes': self.initial_nodes,
            'initial_sub_layers': self.initial_sub_layers,
            'min_nodes': self.min_nodes,
            'max_nodes': self.max_nodes,
            'growth_rate': self.growth_rate,
            'pruning_threshold': self.pruning_threshold,
            'camera_index': self.camera_index,
            'vision_cone_length': self.vision_cone_length,
            'movement_speed': self.movement_speed,
            'max_sub_layers': self.max_sub_layers,
            'depth': self.depth
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class AdaptiveNode:
    def __init__(self, id, state=None, connections=None, position=None):
        self.id = id
        self.state = state if state is not None else np.zeros(10)
        self.connections = connections if connections is not None else {}  # Dict[node_id, weight]
        self.position = position if position is not None else [0.0, 0.0, 0.0]
        self.activation_history = deque(maxlen=100)
        self.success_rate = 1.0
        self.visual_memory = deque(maxlen=50)
        self.response_patterns = {}

    def activate(self, input_signal: np.ndarray):
        """Activate the node based on input signal."""
        self.state = np.tanh(input_signal)
        self.activation_history.append(self.state.copy())

    def adapt(self, neighbor_states: Dict[str, np.ndarray]):
        """
        Adapt connections based on neighbor states using Hebbian learning.
        neighbor_states: Dict[node_id, state_vector]
        """
        for neighbor_id, neighbor_state in neighbor_states.items():
            # Hebbian learning: Δw = η * x * y
            # x: current node's state, y: neighbor node's state
            x = self.state
            y = neighbor_state
            eta = 0.01  # Learning rate
            delta_w = eta * np.outer(x, y)  # Assuming weight matrix
            if neighbor_id in self.connections:
                self.connections[neighbor_id] += delta_w
            else:
                self.connections[neighbor_id] = delta_w
            # Apply weight decay
            self.connections[neighbor_id] *= 0.99  # Prevent unbounded growth

class AdaptiveNetwork:
    def __init__(self, config: SystemConfig, layer_id: str, depth: int = 1):
        self.config = config
        self.layer_id = layer_id
        self.depth = depth
        self.nodes: Dict[str, AdaptiveNode] = {}
        self.sub_layers: List['AdaptiveNetwork'] = []
        self.node_lock = threading.Lock()
        self.initialize_nodes()

        if self.depth < self.config.max_sub_layers:
            # Initialize sub-layers recursively
            for i in range(self.config.initial_sub_layers):
                sub_layer_id = f"{self.layer_id}.{i}"
                sub_layer = AdaptiveNetwork(
                    config=self.config,
                    layer_id=sub_layer_id,
                    depth=self.depth + 1
                )
                self.sub_layers.append(sub_layer)

        self.initialize_connections()

    def initialize_nodes(self):
        """Initialize nodes with random positions."""
        for i in range(self.config.initial_nodes):
            position = (
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            )
            self.nodes[f"{self.layer_id}.{i}"] = AdaptiveNode(
                id=f"{self.layer_id}.{i}",
                position=position
            )
        logging.info(f"Layer {self.layer_id}: Initialized {self.config.initial_nodes} nodes.")

    def initialize_connections(self):
        """Initialize connections between nodes following fractal patterns."""
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                # Example fractal connection pattern: connect nodes with indices differing by powers of 2
                index_diff = j - i
                if (index_diff & (index_diff - 1)) == 0:
                    weight = np.random.uniform(0.5, 1.0)
                    self.nodes[node_ids[i]].connections[node_ids[j]] = weight
                    self.nodes[node_ids[j]].connections[node_ids[i]] = weight
        logging.info(f"Layer {self.layer_id}: Initialized connections among nodes.")

    def process_input(self, input_data: np.ndarray):
        """Process input data and update node states."""
        with self.node_lock:
            for node in self.nodes.values():
                node.activate(input_data)
        # Propagate to sub-layers
        for sub_layer in self.sub_layers:
            sub_layer.process_input(input_data)

    def get_output(self) -> np.ndarray:
        """Generate output from node states."""
        with self.node_lock:
            output = np.mean([node.state for node in self.nodes.values()], axis=0)
        # Aggregate outputs from sub-layers
        for sub_layer in self.sub_layers:
            sub_output = sub_layer.get_output()
            output += sub_output
        return output

    def add_sub_layer(self):
        """Add a sub-layer dynamically."""
        if len(self.sub_layers) < self.config.max_sub_layers:
            sub_layer_id = f"{self.layer_id}.{len(self.sub_layers)}"
            sub_layer = AdaptiveNetwork(
                config=self.config,
                layer_id=sub_layer_id,
                depth=self.depth + 1
            )
            self.sub_layers.append(sub_layer)
            logging.info(f"Layer {self.layer_id}: Added sub-layer {sub_layer_id}.")

    def remove_sub_layer(self):
        """Remove a sub-layer dynamically."""
        if self.sub_layers:
            removed_layer = self.sub_layers.pop()
            logging.info(f"Layer {self.layer_id}: Removed sub-layer {removed_layer.layer_id}.")

class SensoryProcessor:
    def __init__(self, config: SystemConfig, network: AdaptiveNetwork):
        self.config = config
        self.network = network
        self.webcam = cv2.VideoCapture(self.config.camera_index)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open webcam with index {self.config.camera_index}")
        logging.info(f"Webcam with index {self.config.camera_index} opened.")

    def process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract visual features to inform AI movement."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        edges = cv2.Canny(gray, 100, 200)
        motion = np.mean(edges) / 255.0
        return {'motion': motion, 'brightness': brightness}

    def cleanup(self):
        if self.webcam:
            self.webcam.release()
            logging.info("Webcam released.")

class AdaptiveSystem:
    def __init__(self, gui_queue: queue.Queue, vis_queue: queue.Queue, config: SystemConfig):
        self.config = config
        self.network = AdaptiveNetwork(self.config, layer_id="0", depth=1)
        try:
            self.sensory_processor = SensoryProcessor(self.config, self.network)
        except RuntimeError as e:
            messagebox.showerror("Webcam Error", str(e))
            logging.error(f"Failed to initialize SensoryProcessor: {e}")
            self.sensory_processor = None
        self.gui_queue = gui_queue
        self.vis_queue = vis_queue
        self.running = False
        self.capture_thread = None
        self.last_growth_time = time.time()

    def start(self):
        if not self.running and self.sensory_processor is not None:
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            logging.info("Adaptive system started.")

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.sensory_processor:
            self.sensory_processor.cleanup()
        logging.info("Adaptive system stopped.")

    def capture_loop(self):
        while self.running and self.sensory_processor is not None:
            try:
                ret, frame = self.sensory_processor.webcam.read()
                if ret:
                    features = self.sensory_processor.process_frame(frame)
                    dx = (features['brightness'] - 0.5) * 2 * self.config.movement_speed
                    dy = features['motion'] * self.config.movement_speed
                    self.network.process_input(np.array([dx, dy]))

                    # Layer management with proper pruning logic
                    current_time = time.time()
                    if (current_time - self.last_growth_time) > 0.1:  # Check every 100ms
                        # Growth check
                        if np.random.rand() < self.config.growth_rate:
                            self.network.add_sub_layer()
                        
                        # Pruning check - only remove if above threshold
                        if len(self.network.sub_layers) > 0:
                            if np.random.rand() < self.config.pruning_threshold:
                                self.network.remove_sub_layer()
                        
                        # Process Hebbian connections
                        self.network.process_input(np.random.rand(10))
                        self.last_growth_time = current_time

                    # GUI data preparation
                    gui_data = {
                        'frame': frame,
                        'position': self.network.get_output().tolist(),
                        'direction': 0.0
                    }

                    # Visualization data preparation
                    layers_positions = self.collect_layer_positions(self.network)
                    vis_data = {
                        'layers': layers_positions
                    }

                    if not self.gui_queue.full():
                        self.gui_queue.put(gui_data)
                    if not self.vis_queue.full():
                        self.vis_queue.put(vis_data)
            except Exception as e:
                logging.error(f"Error in capture loop: {e}")
            time.sleep(0.01)

    def collect_layer_positions(self, layer: AdaptiveNetwork) -> Dict[str, List[List[float]]]:
        """Recursively collect node positions for each layer."""
        positions = {layer.layer_id: [node.position for node in layer.nodes.values()]}
        for sub_layer in layer.sub_layers:
            positions.update(self.collect_layer_positions(sub_layer))
        return positions

    def save_system(self, filepath: str):
        """Save the system's configuration and node states to a JSON file."""
        try:
            with self.network.node_lock:
                nodes_data = {
                    node_id: {
                        'position': node.position,
                        'state': node.state.tolist(),
                        'connections': {k: v for k, v in node.connections.items()}
                    } for node_id, node in self.network.nodes.items()
                }
            data = {
                'config': self.config.to_dict(),
                'nodes': nodes_data
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"System saved to {filepath}.")
            messagebox.showinfo("Save System", f"System successfully saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save system: {e}")
            messagebox.showerror("Save System", f"Failed to save system: {e}")

    def load_system(self, filepath: str):
        """Load the system's configuration and node states from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Update configuration
            self.config.update_from_dict(data['config'])
            # Update nodes
            with self.network.node_lock:
                self.network.nodes = {}
                for node_id, node_info in data['nodes'].items():
                    node = AdaptiveNode(
                        id=node_id,
                        position=node_info['position'],
                        state=np.array(node_info['state']),
                        connections=node_info['connections']
                    )
                    self.network.nodes[node_id] = node
            logging.info(f"System loaded from {filepath}.")
            messagebox.showinfo("Load System", f"System successfully loaded from {filepath}.")
        except Exception as e:
            logging.error(f"Failed to load system: {e}")
            messagebox.showerror("Load System", f"Failed to load system: {e}")

class FractalNodeVisualizer:
    """Visualization tool for fractal neural networks with layer-specific focus."""
    def __init__(self, parent, network: AdaptiveNetwork, zoom_level=1.0, pan_x=0, pan_y=0, depth=0):
        self.parent = parent
        self.network = network
        self.zoom_level = zoom_level
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.depth = depth
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind('<Button-2>', self._start_pan)
        self.canvas.bind('<B2-Motion>', self._on_pan)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.selected_layer = None  # Currently selected layer for focus
        self.create_layer_selection()
        self.update_visualization()

    def create_layer_selection(self):
        """Create a dropdown for layer selection."""
        control_frame = ttk.Frame(self.parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Focus Layer:").pack(side=tk.LEFT, padx=5)
        self.layer_var = tk.StringVar()
        self.layer_combobox = ttk.Combobox(control_frame, textvariable=self.layer_var, state='readonly')
        self.layer_combobox['values'] = self.get_all_layers()
        if self.layer_combobox['values']:
            self.layer_combobox.current(0)
            self.selected_layer = self.layer_combobox.get()
        self.layer_combobox.bind("<<ComboboxSelected>>", self.on_layer_select)
        self.layer_combobox.pack(side=tk.LEFT, padx=5)

    def get_all_layers(self) -> List[str]:
        """Recursively retrieve all layer IDs in the network."""
        layers = [self.network.layer_id]
        for sub_layer in self.network.sub_layers:
            layers.extend(self._get_sub_layers(sub_layer))
        return layers

    def _get_sub_layers(self, layer: AdaptiveNetwork) -> List[str]:
        """Helper function to retrieve sub-layer IDs."""
        layers = [layer.layer_id]
        for sub_layer in layer.sub_layers:
            layers.extend(self._get_sub_layers(sub_layer))
        return layers

    def on_layer_select(self, event):
        """Handle layer selection changes."""
        self.selected_layer = self.layer_var.get()
        self.update_visualization()

    def _on_mousewheel(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        self.update_visualization()

    def _start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.update_visualization()

    def _on_canvas_resize(self, event):
        self.update_visualization()

    def update_visualization(self):
        """Recursively draw nodes and connections with layer focus."""
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        center_x = width / 2 + self.pan_x
        center_y = height / 2 + self.pan_y

        self._draw_layer(self.network, center_x, center_y, width, height, self.depth)

    def _draw_layer(self, layer: AdaptiveNetwork, center_x, center_y, width, height, depth):
        """Recursively draw a single layer and its sub-layers, focusing on the selected layer."""
        try:
            # Define scaling factors based on depth to avoid overlap
            scaling_factor = self.zoom_level / (depth + 1)

            # Determine if this layer is the selected layer
            is_selected = (layer.layer_id == self.selected_layer)

            # Set color intensity based on selection
            node_color_base = 'red' if is_selected else self._get_layer_color(depth)
            connection_color_base = 'yellow' if is_selected else self._get_connection_color(depth)

            # Draw connections
            for node in layer.nodes.values():
                for connected_id, weight in node.connections.items():
                    target_node = layer.nodes.get(connected_id)
                    if target_node:
                        x1 = center_x + node.position[0] * width / 4 * scaling_factor
                        y1 = center_y + node.position[1] * height / 4 * scaling_factor
                        x2 = center_x + target_node.position[0] * width / 4 * scaling_factor
                        y2 = center_y + target_node.position[1] * height / 4 * scaling_factor
                        color = connection_color_base
                        self.canvas.create_line(
                            x1, y1, x2, y2,
                            fill=color,
                            width=1 if not is_selected else 2
                        )

            # Draw nodes
            for node in layer.nodes.values():
                x = center_x + node.position[0] * width / 4 * scaling_factor
                y = center_y + node.position[1] * height / 4 * scaling_factor
                strength = np.mean(np.abs(node.state))
                radius = max(5, strength * 10 * scaling_factor)
                color = node_color_base
                self.canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color,
                    outline='white',
                    width=2 if is_selected else 1
                )

            # Recursively draw sub-layers
            for idx, sub_layer in enumerate(layer.sub_layers):
                angle = (2 * np.pi / len(layer.sub_layers)) * idx if len(layer.sub_layers) > 0 else 0
                distance = 150 * scaling_factor
                sub_center_x = center_x + np.cos(angle) * distance
                sub_center_y = center_y + np.sin(angle) * distance
                self._draw_layer(sub_layer, sub_center_x, sub_center_y, width, height, depth + 1)

        except Exception as e:
            logging.error(f"Error drawing layer {layer.layer_id}: {e}")

    def _get_layer_color(self, depth):
        """Return a color based on layer depth."""
        # Color gradient from blue (shallow layers) to green (deeper layers)
        green = int(min(255, depth * 30))
        blue = int(max(0, 255 - depth * 30))
        return f'#{0:02x}{green:02x}{blue:02x}'

    def _get_connection_color(self, depth):
        """Return a color based on layer depth for connections."""
        # Color gradient from light gray to dark gray based on depth
        intensity = int(200 - depth * 20)
        intensity = max(50, intensity)  # Minimum intensity
        return f'#{intensity:02x}{intensity:02x}{intensity:02x}'

class FractalMatplotlibVisualizer:
    """Advanced visualization using matplotlib for fractal neural networks."""
    def __init__(self, parent, network: AdaptiveNetwork):
        self.parent = parent
        self.network = network
        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_visualization()

    def update_visualization(self):
        """Render the fractal visualization using matplotlib."""
        self.ax.cla()
        self.ax.set_aspect('auto')
        self.ax.axis('off')

        # Draw nodes and connections using fractal patterns
        for node in self.network.nodes.values():
            x, y, z = node.position
            self.ax.scatter(x, y, z, c='blue', marker='o', s=20, alpha=0.6)
            for connected_id, weight in node.connections.items():
                target_node = self.network.nodes.get(connected_id)
                if target_node:
                    tx, ty, tz = target_node.position
                    self.ax.plot(
                        [x, tx],
                        [y, ty],
                        [z, tz],
                        color=self._get_connection_color(weight),
                        linewidth=0.5
                    )

        # Handle sub-layers recursively
        for sub_layer in self.network.sub_layers:
            self._draw_sub_layer(sub_layer)

        self.canvas.draw()
        self.parent.after(1000, self.update_visualization)  # Update every second

    def _draw_sub_layer(self, layer: AdaptiveNetwork):
        """Recursively draw sub-layers."""
        for node in layer.nodes.values():
            x, y, z = node.position
            self.ax.scatter(x, y, z, c='green', marker='o', s=10, alpha=0.4)
            for connected_id, weight in node.connections.items():
                target_node = layer.nodes.get(connected_id)
                if target_node:
                    tx, ty, tz = target_node.position
                    self.ax.plot(
                        [x, tx],
                        [y, ty],
                        [z, tz],
                        color=self._get_connection_color(weight),
                        linewidth=0.3
                    )

        for sub_sub_layer in layer.sub_layers:
            self._draw_sub_layer(sub_sub_layer)

    def _get_connection_color(self, weight):
        """Return a color based on connection weight."""
        # Color gradient from light gray (low weight) to dark gray (high weight)
        intensity = int(weight * 200 + 55)  # Avoid too dark
        return f'#{intensity:02x}{intensity:02x}{intensity:02x}'

class NodeVisualizer:
    """Separate window for 3D node visualization with layer selection."""
    def __init__(self, parent, vis_queue: queue.Queue, network: AdaptiveNetwork):
        self.parent = parent
        self.vis_queue = vis_queue
        self.network = network
        self.window = tk.Toplevel(parent)
        self.window.title("3D Node Visualization")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.selected_layer = tk.StringVar()
        self.create_widgets()
        self.nodes_positions = {}
        self.update_visualization()

    def create_widgets(self):
        # Layer selection dropdown
        control_frame = ttk.Frame(self.window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Select Layer:").pack(side=tk.LEFT, padx=5)
        self.layer_combobox = ttk.Combobox(control_frame, textvariable=self.selected_layer, state='readonly')
        self.layer_combobox['values'] = self.get_all_layers()
        if self.layer_combobox['values']:
            self.layer_combobox.current(0)
            self.selected_layer.set(self.layer_combobox.get())
        self.layer_combobox.bind("<<ComboboxSelected>>", self.on_layer_select)
        self.layer_combobox.pack(side=tk.LEFT, padx=5)
        
        # Create a matplotlib figure
        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Adaptive Network Nodes")
    
        # Embed the figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def get_all_layers(self) -> List[str]:
        """Recursively retrieve all layer IDs in the network."""
        layers = [self.network.layer_id]
        for sub_layer in self.network.sub_layers:
            layers.extend(self._get_sub_layers(sub_layer))
        return layers

    def _get_sub_layers(self, layer: AdaptiveNetwork) -> List[str]:
        """Helper function to retrieve sub-layer IDs."""
        layers = [layer.layer_id]
        for sub_layer in layer.sub_layers:
            layers.extend(self._get_sub_layers(sub_layer))
        return layers

    def on_layer_select(self, event):
        """Handle layer selection changes."""
        self.selected_layer.set(self.layer_combobox.get())
        self.update_visualization()
    
    def update_visualization(self):
        """Render the 3D visualization based on selected layer."""
        try:
            while not self.vis_queue.empty():
                data = self.vis_queue.get_nowait()
                if 'layers' in data:
                    self.layers_data = data['layers']
                    logging.info(f"NodeVisualizer received data for {len(self.layers_data)} layers.")
    
            self.ax.cla()  # Clear the current axes
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            self.ax.set_zlim([-2, 2])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title("Adaptive Network Nodes")
    
            # Get selected layer
            selected = self.selected_layer.get()
            if not selected:
                return
    
            # Extract positions for the selected layer
            positions = self.layers_data.get(selected, [])
            if positions:
                xs, ys, zs = zip(*positions)
                # Normalize positions for visualization
                xs_norm = [(x - min(xs)) / (max(xs) - min(xs) + 1e-5) * 4 - 2 for x in xs]
                ys_norm = [(y - min(ys)) / (max(ys) - min(ys) + 1e-5) * 4 - 2 for y in ys]
                zs_norm = [(z - min(zs)) / (max(zs) - min(zs) + 1e-5) * 4 - 2 for z in zs]
                # Plot nodes
                self.ax.scatter(xs_norm, ys_norm, zs_norm, c='b', marker='o', s=20, alpha=0.6)
                logging.info(f"Plotted {len(xs_norm)} nodes for layer {selected}.")
            else:
                logging.info(f"No nodes to plot for layer {selected}.")
    
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error in node visualization update: {e}")
        finally:
            self.window.after(100, self.update_visualization)  # Update every 100 ms

    def on_close(self):
        self.window.destroy()

class ConfigWindow:
    """Configuration window for adjusting system parameters."""
    def __init__(self, parent, config: SystemConfig, adaptive_system: AdaptiveSystem):
        self.parent = parent
        self.config = config
        self.adaptive_system = adaptive_system
        self.window = tk.Toplevel(parent)
        self.window.title("Configuration")
        self.window.geometry("400x400")
        self.window.resizable(False, False)
        self.window.grab_set()  # Make the config window modal
        self.create_widgets()

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 5}

        # Depth
        ttk.Label(self.window, text="Depth:").grid(row=0, column=0, sticky=tk.W, **padding)
        self.depth_var = tk.IntVar(value=self.config.depth)
        self.depth_spinbox = ttk.Spinbox(self.window, from_=1, to=10, textvariable=self.depth_var, width=10)
        self.depth_spinbox.grid(row=0, column=1, **padding)

        # Pruning Rate
        ttk.Label(self.window, text="Pruning Threshold:").grid(row=1, column=0, sticky=tk.W, **padding)
        self.pruning_rate_var = tk.DoubleVar(value=self.config.pruning_threshold)
        self.pruning_rate_entry = ttk.Entry(self.window, textvariable=self.pruning_rate_var, width=12)
        self.pruning_rate_entry.grid(row=1, column=1, **padding)

        # Growth Rate
        ttk.Label(self.window, text="Growth Rate:").grid(row=2, column=0, sticky=tk.W, **padding)
        self.growth_rate_var = tk.DoubleVar(value=self.config.growth_rate)
        self.growth_rate_entry = ttk.Entry(self.window, textvariable=self.growth_rate_var, width=12)
        self.growth_rate_entry.grid(row=2, column=1, **padding)

        # Minimum Nodes
        ttk.Label(self.window, text="Minimum Nodes:").grid(row=3, column=0, sticky=tk.W, **padding)
        self.min_nodes_var = tk.IntVar(value=self.config.min_nodes)
        self.min_nodes_spinbox = ttk.Spinbox(
            self.window, from_=1, to=self.config.max_nodes, textvariable=self.min_nodes_var, width=10
        )
        self.min_nodes_spinbox.grid(row=3, column=1, **padding)

        # Maximum Nodes
        ttk.Label(self.window, text="Maximum Nodes:").grid(row=4, column=0, sticky=tk.W, **padding)
        self.max_nodes_var = tk.IntVar(value=self.config.max_nodes)
        self.max_nodes_spinbox = ttk.Spinbox(
            self.window, from_=self.config.min_nodes, to=10000, textvariable=self.max_nodes_var, width=10
        )
        self.max_nodes_spinbox.grid(row=4, column=1, **padding)

        # Webcam Selection
        ttk.Label(self.window, text="Webcam:").grid(row=5, column=0, sticky=tk.W, **padding)
        self.webcam_var = tk.IntVar(value=self.config.camera_index)
        self.webcam_combobox = ttk.Combobox(self.window, textvariable=self.webcam_var, state='readonly', width=8)
        self.webcam_combobox['values'] = self.detect_webcams()
        # Set current selection based on camera_index
        camera_str = str(self.config.camera_index)
        if camera_str in self.webcam_combobox['values']:
            self.webcam_combobox.current(self.webcam_combobox['values'].index(camera_str))
        else:
            self.webcam_combobox.current(0)
        self.webcam_combobox.grid(row=5, column=1, **padding)

        # Save and Load Buttons
        self.save_button = ttk.Button(self.window, text="Save Configuration", command=self.save_configuration)
        self.save_button.grid(row=6, column=0, **padding)

        self.load_button = ttk.Button(self.window, text="Load Configuration", command=self.load_configuration)
        self.load_button.grid(row=6, column=1, **padding)

        # Apply Button
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_changes)
        self.apply_button.grid(row=7, column=0, columnspan=2, pady=20)

    def detect_webcams(self) -> List[str]:
        """Detect available webcams and return their indices as strings."""
        available_cameras = []
        max_tested = 5
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
        if not available_cameras:
            available_cameras.append("0")  # Default to 0 if no cameras found
        return available_cameras

    def save_configuration(self):
        """Save the current configuration and node states to a JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save System Configuration"
        )
        if filepath:
            self.adaptive_system.save_system(filepath)

    def load_configuration(self):
        """Load configuration and node states from a JSON file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load System Configuration"
        )
        if filepath:
            self.adaptive_system.load_system(filepath)
            # Update GUI elements with loaded configuration
            self.depth_var.set(self.config.depth)
            self.pruning_rate_var.set(self.config.pruning_threshold)
            self.growth_rate_var.set(self.config.growth_rate)
            self.min_nodes_var.set(self.config.min_nodes)
            self.max_nodes_var.set(self.config.max_nodes)
            camera_str = str(self.config.camera_index)
            if camera_str in self.webcam_combobox['values']:
                self.webcam_combobox.current(self.webcam_combobox['values'].index(camera_str))
            else:
                self.webcam_combobox.current(0)

    def apply_changes(self):
        """Apply the changes made in the configuration window."""
        try:
            # Retrieve values from the GUI
            new_depth = self.depth_var.get()
            new_pruning_rate = float(self.pruning_rate_var.get())
            new_growth_rate = float(self.growth_rate_var.get())
            new_min_nodes = self.min_nodes_var.get()
            new_max_nodes = self.max_nodes_var.get()
            new_camera_index = int(self.webcam_var.get())

            # Validate values
            if new_min_nodes > new_max_nodes:
                messagebox.showerror("Configuration Error", "Minimum nodes cannot exceed maximum nodes.")
                return

            # Update configuration
            self.config.depth = new_depth
            self.config.pruning_threshold = new_pruning_rate
            self.config.growth_rate = new_growth_rate
            self.config.min_nodes = new_min_nodes
            self.config.max_nodes = new_max_nodes
            self.config.camera_index = new_camera_index

            # Apply webcam change
            was_running = self.adaptive_system.running
            self.adaptive_system.stop()
            try:
                # Update webcam in sensory processor
                self.adaptive_system.config.camera_index = new_camera_index
                self.adaptive_system.sensory_processor = SensoryProcessor(self.adaptive_system.config, self.adaptive_system.network)
                if was_running:
                    self.adaptive_system.start()
            except RuntimeError as e:
                messagebox.showerror("Webcam Error", str(e))
                logging.error(f"Failed to change webcam: {e}")
                return

            messagebox.showinfo("Configuration", "Configuration applied successfully.")
            self.window.destroy()
        except Exception as e:
            logging.error(f"Error applying configuration: {e}")
            messagebox.showerror("Configuration Error", f"Failed to apply configuration: {e}")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractal Adaptive System")
        self.root.geometry("1200x800")
        self.gui_queue = queue.Queue(maxsize=50)
        self.vis_queue = queue.Queue(maxsize=50)
        self.config = SystemConfig()
        self.system = AdaptiveSystem(self.gui_queue, self.vis_queue, self.config)
        self.node_visualizer = None  # Will hold the NodeVisualizer instance
        self.create_widgets()
        self.update_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Create menu bar without camera selection to avoid conflicts
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        # Add Node Visualization Button
        self.visualize_button = ttk.Button(control_frame, text="Visualize Nodes", command=self.open_node_visualizer)
        self.visualize_button.pack(side=tk.LEFT, padx=5)

        # Add Config Button
        self.config_button = ttk.Button(control_frame, text="Config", command=self.open_config_window)
        self.config_button.pack(side=tk.LEFT, padx=5)

        # Add Save and Load Buttons
        self.save_button = ttk.Button(control_frame, text="Save", command=self.save_system)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(control_frame, text="Load", command=self.load_system)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Canvas for video feed
        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def save_system(self):
        """Save the system's configuration and node states."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save System"
        )
        if filepath:
            self.system.save_system(filepath)

    def load_system(self):
        """Load the system's configuration and node states."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load System"
        )
        if filepath:
            self.system.load_system(filepath)

    def open_config_window(self):
        """Open the configuration window."""
        ConfigWindow(self.root, self.config, self.system)

    def _on_canvas_resize(self, event):
        self.system.config.display_width = event.width
        self.system.config.display_height = event.height

    def update_gui(self):
        try:
            while not self.gui_queue.empty():
                data = self.gui_queue.get_nowait()
                if 'frame' in data and data['frame'] is not None:
                    frame = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas._photo = photo  # Keep a reference to prevent garbage collection

                    if 'position' in data:
                        x, y = data['position'][:2]  # Assuming position is 2D for visualization
                        direction = data.get('direction', 0)
                        cone_length = self.system.config.vision_cone_length
                        cone_angle = np.pi / 4
                        p1 = (x, y)
                        p2 = (
                            x + cone_length * np.cos(direction - cone_angle),
                            y + cone_length * np.sin(direction - cone_angle)
                        )
                        p3 = (
                            x + cone_length * np.cos(direction + cone_angle),
                            y + cone_length * np.sin(direction + cone_angle)
                        )
                        self.canvas.create_polygon(
                            p1[0], p1[1], p2[0], p2[1], p3[0], p3[1],
                            fill='#00ff00', stipple='gray50', outline='#00ff00', width=2
                        )
                        radius = 10
                        self.canvas.create_oval(
                            x - radius, y - radius, x + radius, y + radius,
                            fill='#00ff00', outline='white', width=2
                        )
        except Exception as e:
            logging.error(f"Error updating GUI: {e}")

        self.root.after(33, self.update_gui)  # Approximately 30 FPS

    def start(self):
        self.system.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        logging.info("System started via GUI.")

    def stop(self):
        self.system.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        logging.info("System stopped via GUI.")

    def open_node_visualizer(self):
        if self.node_visualizer is None or not tk.Toplevel.winfo_exists(self.node_visualizer.window):
            self.node_visualizer = NodeVisualizer(self.root, self.vis_queue, self.system.network)
            logging.info("Node visualization window opened.")
        else:
            self.node_visualizer.window.lift()  # Bring to front if already open

    def on_close(self):
        if self.system.running:
            self.stop()
        if self.node_visualizer and tk.Toplevel.winfo_exists(self.node_visualizer.window):
            self.node_visualizer.window.destroy()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
