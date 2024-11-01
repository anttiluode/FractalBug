import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import sys
import threading
import queue
import logging
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
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
        logging.FileHandler("hivemind_system.log"),
        logging.StreamHandler()
    ]
)

class SystemConfig:
    def __init__(self):
        # Display settings
        self.display_width = 800
        self.display_height = 600
        
        # Network architecture
        self.initial_nodes = 20
        self.initial_sub_layers = 2
        self.min_nodes = 10
        self.max_nodes = 500
        self.max_sub_layers = 4
        self.encoding_dimension = 32
        
        # Growth and adaptation
        self.growth_rate = 0.1
        self.pruning_threshold = 0.3
        self.learning_rate = 0.01
        self.adaptation_rate = 0.1
        
        # Movement and behavior
        self.movement_speed = 3.0
        self.rotation_speed = 0.1
        self.vision_cone_length = 150
        self.vision_cone_angle = np.pi / 4
        
        # Memory and temporal
        self.memory_size = 100
        self.temporal_decay = 0.9
        
        # Camera settings
        self.camera_index = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class BidirectionalNode:
    def __init__(self, id: str, state: Optional[np.ndarray] = None, position: Optional[List[float]] = None):
        self.id = id
        self.state = state if state is not None else np.zeros(10)
        self.position = position if position is not None else [0.0, 0.0, 0.0]
        
        # Encoding vectors
        self.forward_encoding: Optional[np.ndarray] = None
        self.backward_encoding: Optional[np.ndarray] = None
        
        # Movement and behavior
        self.movement_influence = np.zeros(3)  # (dx, dy, rotation)
        self.success_rate = 1.0
        
        # Memory and history
        self.activation_history = deque(maxlen=100)
        self.visual_memory = deque(maxlen=50)
        self.movement_history = deque(maxlen=100)
        
        # Connections to other nodes
        self.connections: Dict[str, float] = {}
        self.connection_strengths = {}

    def compute_bidirectional_encoding(
        self, 
        depth: int, 
        max_depth: int, 
        temporal_position: float,
        encoding_dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both forward and backward positional encodings.
        
        Args:
            depth: Current layer depth
            max_depth: Maximum network depth
            temporal_position: Position in temporal sequence (0 to 1)
            encoding_dim: Dimension of encoding vectors
            
        Returns:
            Tuple of (forward_encoding, backward_encoding)
        """
        self.forward_encoding = np.zeros((encoding_dim,))
        self.backward_encoding = np.zeros((encoding_dim,))
        
        # Compute normalized positions
        forward_position = float(depth) / max_depth + temporal_position
        backward_position = float(max_depth - depth) / max_depth + (1 - temporal_position)
        
        # Generate sinusoidal encodings
        for i in range(0, encoding_dim, 2):
            denominator = np.power(10000, 2 * i / encoding_dim)
            
            # Forward encoding
            self.forward_encoding[i] = np.sin(forward_position / denominator)
            if i + 1 < encoding_dim:
                self.forward_encoding[i + 1] = np.cos(forward_position / denominator)
            
            # Backward encoding
            self.backward_encoding[i] = np.sin(backward_position / denominator)
            if i + 1 < encoding_dim:
                self.backward_encoding[i + 1] = np.cos(backward_position / denominator)
        
        return self.forward_encoding, self.backward_encoding

    def update_state(self, input_signal: np.ndarray, temporal_context: float = 0.0):
        """
        Update node state based on input signal and temporal context.
        
        Args:
            input_signal: Input vector
            temporal_context: Temporal position (0 to 1)
        """
        # Combine input with encodings
        if self.forward_encoding is not None and self.backward_encoding is not None:
            combined_input = np.concatenate([
                input_signal,
                self.forward_encoding * temporal_context,
                self.backward_encoding * (1 - temporal_context)
            ])
        else:
            combined_input = input_signal
            
        # Apply activation function
        self.state = np.tanh(combined_input)
        self.activation_history.append(self.state.copy())

    def adapt_connections(self, neighbor_states: Dict[str, np.ndarray], learning_rate: float):
        """
        Adapt connections to neighbors using Hebbian learning.
        
        Args:
            neighbor_states: Dictionary of neighbor IDs to their states
            learning_rate: Learning rate for adaptation
        """
        for neighbor_id, neighbor_state in neighbor_states.items():
            if neighbor_id not in self.connections:
                self.connections[neighbor_id] = 0.0
                
            # Hebbian update
            self.connections[neighbor_id] += (
                learning_rate * 
                np.dot(self.state, neighbor_state) * 
                (1 - abs(self.connections[neighbor_id]))
            )
            
            # Apply weight decay
            self.connections[neighbor_id] *= 0.99

    def update_movement_influence(self, 
                                visual_input: np.ndarray, 
                                temporal_context: float,
                                success_rate: float):
        """
        Update node's movement influence based on inputs and success rate.
        
        Args:
            visual_input: Visual input vector
            temporal_context: Temporal position (0 to 1)
            success_rate: Rate of successful movements
        """
        # Combine inputs for movement computation
        combined_features = np.concatenate([
            visual_input,
            self.forward_encoding if self.forward_encoding is not None else np.zeros(10),
            self.backward_encoding if self.backward_encoding is not None else np.zeros(10),
            self.position
        ])
        
        # Simple neural network processing
        W1 = np.random.randn(combined_features.shape[0], 32) * 0.1
        W2 = np.random.randn(32, 3) * 0.1
        
        hidden = np.tanh(combined_features @ W1)
        movement = np.tanh(hidden @ W2)
        
        # Scale movement by success rate and temporal context
        self.movement_influence = (
            movement * 
            success_rate * 
            (temporal_context * 0.5 + 0.5)  # Ensure some baseline influence
        )
        
        self.movement_history.append(self.movement_influence.copy())

# end of claude part 1 

class HiveMindNetwork:
    def __init__(self, config: SystemConfig, layer_id: str, depth: int = 1):
        self.config = config
        self.layer_id = layer_id
        self.depth = depth
        self.nodes: Dict[str, BidirectionalNode] = {}
        self.sub_layers: List['HiveMindNetwork'] = []
        self.node_lock = threading.Lock()
        
        # Collective memory and state
        self.collective_memory = deque(maxlen=self.config.memory_size)
        self.layer_movement = np.zeros(3)  # (dx, dy, rotation)
        self.success_history = deque(maxlen=100)
        self.mean_success_rate = 1.0
        
        # Layer state
        self.temporal_position = 0.0
        self.layer_state = np.zeros(self.config.encoding_dimension)
        
        self.initialize_nodes()
        self.initialize_sublayers()

    def initialize_nodes(self):
        """Initialize nodes with positions following fractal patterns."""
        for i in range(self.config.initial_nodes):
            # Generate fractal position using 3D Cantor set inspired coordinates
            position = self._generate_fractal_position(i)
            
            node = BidirectionalNode(
                id=f"{self.layer_id}.{i}",
                position=position
            )
            self.nodes[node.id] = node
            
            # Initialize encodings
            node.compute_bidirectional_encoding(
                self.depth,
                self.config.max_sub_layers,
                self.temporal_position,
                self.config.encoding_dimension
            )
        
        logging.info(f"Layer {self.layer_id}: Initialized {len(self.nodes)} nodes")

    def _generate_fractal_position(self, index: int) -> List[float]:
        """Generate position using fractal patterns."""
        # Use bit patterns of index for fractal-like distribution
        x = 0.0
        y = 0.0
        z = 0.0
        scale = 1.0
        
        for i in range(8):  # Use 8 iterations for detail
            bit_x = (index >> (i * 3)) & 1
            bit_y = (index >> (i * 3 + 1)) & 1
            bit_z = (index >> (i * 3 + 2)) & 1
            
            x += (bit_x - 0.5) * scale
            y += (bit_y - 0.5) * scale
            z += (bit_z - 0.5) * scale
            scale *= 0.5
            
        return [x, y, z]

    def initialize_sublayers(self):
        """Initialize sublayers if not at maximum depth."""
        if self.depth < self.config.max_sub_layers:
            for i in range(self.config.initial_sub_layers):
                sub_layer_id = f"{self.layer_id}.{i}"
                sub_layer = HiveMindNetwork(
                    config=self.config,
                    layer_id=sub_layer_id,
                    depth=self.depth + 1
                )
                self.sub_layers.append(sub_layer)

    def process_input(self, visual_input: np.ndarray, reward: float = 0.0):
        """Process input through the network and update states."""
        with self.node_lock:
            # Update temporal position
            self.temporal_position = len(self.collective_memory) / self.config.memory_size
            
            # Update node states and encodings
            for node in self.nodes.values():
                # Compute new encodings
                node.compute_bidirectional_encoding(
                    self.depth,
                    self.config.max_sub_layers,
                    self.temporal_position,
                    self.config.encoding_dimension
                )
                
                # Update node state
                node.update_state(visual_input, self.temporal_position)
                
                # Update movement influence
                node.update_movement_influence(
                    visual_input,
                    self.temporal_position,
                    self.mean_success_rate
                )
            
            # Process neighbor interactions
            self._process_neighbor_interactions()
            
            # Update layer state
            self.layer_state = np.mean([node.state for node in self.nodes.values()], axis=0)
            
            # Store in collective memory
            self.collective_memory.append(self.layer_state)
            
            # Process reward
            if reward != 0.0:
                self.success_history.append(1.0 if reward > 0 else 0.0)
                self.mean_success_rate = np.mean(list(self.success_history))
            
            # Propagate to sublayers
            for sub_layer in self.sub_layers:
                sub_layer.process_input(visual_input, reward * self.config.temporal_decay)

    def _process_neighbor_interactions(self):
        """Process interactions between neighboring nodes."""
        for node in self.nodes.values():
            # Get states of connected neighbors
            neighbor_states = {
                neighbor_id: self.nodes[neighbor_id].state
                for neighbor_id in node.connections.keys()
                if neighbor_id in self.nodes
            }
            
            # Adapt connections using Hebbian learning
            node.adapt_connections(neighbor_states, self.config.learning_rate)

    def compute_collective_movement(self) -> np.ndarray:
        """Compute collective movement decision from all layers."""
        with self.node_lock:
            # Compute this layer's movement influence
            layer_movement = np.zeros(3)
            for node in self.nodes.values():
                layer_movement += node.movement_influence
            
            # Normalize layer movement
            if np.any(layer_movement != 0):
                layer_movement = layer_movement / np.linalg.norm(layer_movement)
            
            # Collect movements from sublayers
            all_movements = [layer_movement]
            weights = [1.0]  # Base layer weight
            
            for sublayer in self.sub_layers:
                sublayer_movement = sublayer.compute_collective_movement()
                # Weight sublayer contributions based on depth and success
                weight = (self.config.temporal_decay ** sublayer.depth) * sublayer.mean_success_rate
                all_movements.append(sublayer_movement)
                weights.append(weight)
            
            # Compute weighted average
            weights = np.array(weights) / sum(weights)
            collective_movement = np.zeros(3)
            for movement, weight in zip(all_movements, weights):
                collective_movement += movement * weight
            
            # Store as layer movement
            self.layer_movement = collective_movement
            return collective_movement

    def adapt_to_feedback(self, reward: float):
        """Adapt network based on movement success."""
        with self.node_lock:
            # Update success history
            self.success_history.append(1.0 if reward > 0 else 0.0)
            self.mean_success_rate = np.mean(list(self.success_history))
            
            # Adapt node positions and connections
            for node in self.nodes.values():
                if reward > 0:
                    # Reinforce successful movement patterns
                    node.position = [
                        p + m * reward * self.config.adaptation_rate
                        for p, m in zip(node.position, node.movement_influence)
                    ]
                else:
                    # Reduce influence of unsuccessful patterns
                    node.movement_influence *= (1.0 + reward)
            
            # Propagate adaptation to sublayers
            for sublayer in self.sub_layers:
                sublayer.adapt_to_feedback(reward * self.config.temporal_decay)

    def get_output(self) -> np.ndarray:
        """Get network output by combining all layer states."""
        with self.node_lock:
            output = self.layer_state
            for sublayer in self.sub_layers:
                output += sublayer.get_output() * (self.config.temporal_decay ** sublayer.depth)
            return output

# end of claude part 2 

class SensoryProcessor:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.webcam = cv2.VideoCapture(self.config.camera_index)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open webcam with index {self.config.camera_index}")
        
        self.frame_history = deque(maxlen=10)
        self.motion_history = deque(maxlen=10)
        self.feature_history = deque(maxlen=10)
        
        # Initialize feature extraction
        self.feature_extractor = cv2.SIFT_create()
        
        logging.info(f"Initialized SensoryProcessor with camera index {self.config.camera_index}")

    def process_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract visual features and compute motion analysis.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing processed features
        """
        # Store frame in history
        self.frame_history.append(frame.copy())
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT features
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.zeros((1, 128))  # Default SIFT descriptor size
        
        # Compute basic statistics
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        
        # Compute motion if we have previous frames
        motion = np.zeros_like(gray, dtype=np.float32)
        if len(self.frame_history) > 1:
            prev_gray = cv2.cvtColor(self.frame_history[-2], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            self.motion_history.append(motion)
        
        # Compute motion direction
        motion_direction = 0.0
        if len(self.motion_history) > 0:
            motion_mean = np.mean(motion)
            if motion_mean > 0.01:  # Threshold for significant motion
                motion_direction = np.arctan2(
                    np.mean(flow[..., 1]),
                    np.mean(flow[..., 0])
                )
        
        # Create feature vector
        feature_vector = np.concatenate([
            [brightness, contrast],
            np.mean(descriptors, axis=0) if len(descriptors.shape) > 1 else descriptors,
            [motion_direction]
        ])
        
        self.feature_history.append(feature_vector)
        
        return {
            'feature_vector': feature_vector,
            'motion': motion,
            'motion_direction': motion_direction,
            'brightness': brightness,
            'contrast': contrast
        }

    def cleanup(self):
        """Release webcam resources."""
        if self.webcam:
            self.webcam.release()
            logging.info("Released webcam resources")

class HiveMindSystem:
    def __init__(self, gui_queue: queue.Queue, vis_queue: queue.Queue, config: SystemConfig):
        self.config = config
        self.network = HiveMindNetwork(self.config, layer_id="0", depth=1)
        
        try:
            self.sensory_processor = SensoryProcessor(self.config)
        except RuntimeError as e:
            messagebox.showerror("Webcam Error", str(e))
            logging.error(f"Failed to initialize SensoryProcessor: {e}")
            self.sensory_processor = None
            
        self.gui_queue = gui_queue
        self.vis_queue = vis_queue
        self.running = False
        self.capture_thread = None
        
        # System state
        self.current_position = np.zeros(3)  # (x, y, rotation)
        self.target_position = np.zeros(3)
        self.movement_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        logging.info("Initialized HiveMindSystem")

    def start(self):
        """Start the system processing loop."""
        if not self.running and self.sensory_processor is not None:
            self.running = True
            self.capture_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.capture_thread.start()
            logging.info("Started HiveMindSystem processing loop")

    def stop(self):
        """Stop the system processing loop."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.sensory_processor:
            self.sensory_processor.cleanup()
        logging.info("Stopped HiveMindSystem")

    def processing_loop(self):
        """Main processing loop for the system."""
        while self.running and self.sensory_processor is not None:
            try:
                # Capture and process frame
                ret, frame = self.sensory_processor.webcam.read()
                if not ret:
                    continue
                
                # Extract features
                features = self.sensory_processor.process_frame(frame)
                
                # Process through network
                self.network.process_input(features['feature_vector'])
                
                # Compute collective movement
                movement = self.network.compute_collective_movement()
                
                # Update positions
                new_position = self._compute_new_position(movement)
                reward = self._evaluate_movement(new_position)
                
                # Apply movement if valid
                if reward >= 0:
                    self.current_position = new_position
                
                # Adapt network based on reward
                self.network.adapt_to_feedback(reward)
                
                # Store history
                self.movement_history.append(movement)
                self.reward_history.append(reward)
                
                # Prepare visualization data
                self._prepare_visualization_data(frame, features)
                
            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                
            time.sleep(0.01)

    def _compute_new_position(self, movement: np.ndarray) -> np.ndarray:
        """Compute new position based on movement vector."""
        dx, dy, drotation = movement
        
        # Scale movements
        dx *= self.config.movement_speed
        dy *= self.config.movement_speed
        drotation *= self.config.rotation_speed
        
        # Current position components
        x, y, rotation = self.current_position
        
        # Update position using rotation
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        
        new_x = x + (dx * cos_rot - dy * sin_rot)
        new_y = y + (dx * sin_rot + dy * cos_rot)
        new_rotation = (rotation + drotation) % (2 * np.pi)
        
        return np.array([new_x, new_y, new_rotation])

    def _evaluate_movement(self, new_position: np.ndarray) -> float:
        """Evaluate validity and success of movement."""
        x, y, _ = new_position
        
        # Check boundaries
        if (x < 0 or x >= self.config.display_width or 
            y < 0 or y >= self.config.display_height):
            return -1.0
        
        # Compute reward based on movement goals
        reward = 0.0
        
        # Reward exploration of new areas
        if len(self.movement_history) > 0:
            prev_positions = np.array([(m[0], m[1]) for m in self.movement_history])
            distances = np.sqrt(np.sum((prev_positions - np.array([x, y]))**2, axis=1))
            exploration_reward = np.min(distances) * 0.01
            reward += exploration_reward
        
        # Reward smooth movements
        if len(self.movement_history) > 1:
            prev_movement = self.movement_history[-1]
            movement_smoothness = np.dot(new_position - self.current_position, prev_movement)
            reward += movement_smoothness * 0.1
        
        return reward

    def _prepare_visualization_data(self, frame: np.ndarray, features: Dict[str, Any]):
        """Prepare data for visualization."""
        try:
            # GUI data
            gui_data = {
                'frame': frame,
                'position': self.current_position,
                'movement': self.movement_history[-1] if self.movement_history else np.zeros(3),
                'features': features
            }
            
            # Visualization data
            vis_data = {
                'network_state': self._collect_network_state(),
                'movement_history': list(self.movement_history),
                'reward_history': list(self.reward_history)
            }
            
            # Add to queues if not full
            if not self.gui_queue.full():
                self.gui_queue.put(gui_data)
            if not self.vis_queue.full():
                self.vis_queue.put(vis_data)
                
        except Exception as e:
            logging.error(f"Error preparing visualization data: {e}")

    def _collect_network_state(self) -> Dict[str, Any]:
        """Collect current network state for visualization."""
        def collect_layer_data(layer: HiveMindNetwork) -> Dict[str, Any]:
            layer_data = {
                'id': layer.layer_id,
                'depth': layer.depth,
                'nodes': [
                    {
                        'id': node.id,
                        'position': node.position,
                        'state': node.state.tolist(),
                        'movement_influence': node.movement_influence.tolist()
                    }
                    for node in layer.nodes.values()
                ],
                'sublayers': [
                    collect_layer_data(sublayer)
                    for sublayer in layer.sub_layers
                ]
            }
            return layer_data
            
        return collect_layer_data(self.network)

    def save_system(self, filepath: str):
        """Save system state to file."""
        try:
            state_data = {
                'config': self.config.to_dict(),
                'network_state': self._collect_network_state(),
                'current_position': self.current_position.tolist()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=4)
                
            logging.info(f"System state saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving system state: {e}")
            raise

    def load_system(self, filepath: str):
        """Load system state from file."""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
                
            # Update configuration
            self.config.update_from_dict(state_data['config'])
            
            # Recreate network with loaded state
            self.network = HiveMindNetwork(self.config, layer_id="0", depth=1)
            # TODO: Implement detailed network state restoration
            
            # Restore position
            self.current_position = np.array(state_data['current_position'])
            
            logging.info(f"System state loaded from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading system state: {e}")
            raise

class HiveMindVisualizer:
    """Advanced visualization window for the hive mind network."""
    def __init__(self, parent, vis_queue: queue.Queue, network: HiveMindNetwork):
        self.parent = parent
        self.vis_queue = vis_queue
        self.network = network
        
        self.window = tk.Toplevel(parent)
        self.window.title("Hive Mind Network Visualization")
        self.window.geometry("1200x800")
        
        # Setup tabs for different visualizations
        self.setup_tabs()
        self.create_controls()
        
        # Start update loop
        self.update_visualization()
        
    def setup_tabs(self):
        """Create tabbed interface for different visualizations."""
        self.tab_control = ttk.Notebook(self.window)
        
        # Create tabs
        self.network_tab = ttk.Frame(self.tab_control)
        self.movement_tab = ttk.Frame(self.tab_control)
        self.state_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.network_tab, text='Network Structure')
        self.tab_control.add(self.movement_tab, text='Movement Patterns')
        self.tab_control.add(self.state_tab, text='State Analysis')
        
        self.tab_control.pack(expand=1, fill='both')
        
        # Setup individual visualizations
        self.setup_network_view()
        self.setup_movement_view()
        self.setup_state_view()

    def create_controls(self):
        """Create control panel for visualization options."""
        control_frame = ttk.Frame(self.window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Layer selection
        ttk.Label(control_frame, text="Focus Layer:").pack(side=tk.LEFT, padx=5)
        self.layer_var = tk.StringVar()
        self.layer_combobox = ttk.Combobox(
            control_frame, 
            textvariable=self.layer_var,
            state='readonly'
        )
        self.layer_combobox['values'] = self.get_all_layers()
        self.layer_combobox.pack(side=tk.LEFT, padx=5)
        self.layer_combobox.bind('<<ComboboxSelected>>', self.on_layer_select)
        
        # Visualization options
        self.show_connections = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Show Connections",
            variable=self.show_connections,
            command=self.update_visualization
        ).pack(side=tk.LEFT, padx=5)
        
        self.show_movement = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Show Movement",
            variable=self.show_movement,
            command=self.update_visualization
        ).pack(side=tk.LEFT, padx=5)

    def setup_network_view(self):
        """Setup 3D network visualization."""
        self.network_fig = plt.Figure(figsize=(8, 6))
        self.network_ax = self.network_fig.add_subplot(111, projection='3d')
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, self.network_tab)
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_movement_view(self):
        """Setup movement pattern visualization."""
        self.movement_fig = plt.Figure(figsize=(8, 6))
        self.movement_ax = self.movement_fig.add_subplot(111)
        self.movement_canvas = FigureCanvasTkAgg(self.movement_fig, self.movement_tab)
        self.movement_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_state_view(self):
        """Setup state analysis visualization."""
        self.state_fig = plt.Figure(figsize=(8, 6))
        gs = self.state_fig.add_gridspec(2, 2)
        
        # Create subplots for different state aspects
        self.state_axes = {
            'activation': self.state_fig.add_subplot(gs[0, 0]),
            'encoding': self.state_fig.add_subplot(gs[0, 1]),
            'movement': self.state_fig.add_subplot(gs[1, 0]),
            'success': self.state_fig.add_subplot(gs[1, 1])
        }
        
        self.state_canvas = FigureCanvasTkAgg(self.state_fig, self.state_tab)
        self.state_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_visualization(self):
        """Update all visualizations with current data."""
        try:
            while not self.vis_queue.empty():
                data = self.vis_queue.get_nowait()
                if 'network_state' in data:
                    self.update_network_view(data['network_state'])
                if 'movement_history' in data:
                    self.update_movement_view(data['movement_history'])
                if 'reward_history' in data:
                    self.update_state_analysis(data)

            self.window.after(100, self.update_visualization)
            
        except Exception as e:
            logging.error(f"Error updating visualization: {e}")

    def update_network_view(self, network_state):
        """Update 3D network visualization."""
        self.network_ax.clear()
        
        def draw_layer(layer_data, parent_pos=None):
            positions = np.array([node['position'] for node in layer_data['nodes']])
            
            # Draw nodes
            self.network_ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=self.get_layer_color(layer_data['depth']),
                s=50,
                alpha=0.6
            )
            
            # Draw connections if enabled
            if self.show_connections.get():
                for i, node in enumerate(layer_data['nodes']):
                    # Draw connections between nodes in the same layer
                    for j in range(i + 1, len(layer_data['nodes'])):
                        self.network_ax.plot(
                            [positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            [positions[i, 2], positions[j, 2]],
                            'gray',
                            alpha=0.2
                        )
                    
                    # Draw connections to parent layer
                    if parent_pos is not None:
                        self.network_ax.plot(
                            [positions[i, 0], parent_pos[0]],
                            [positions[i, 1], parent_pos[1]],
                            [positions[i, 2], parent_pos[2]],
                            'gray',
                            alpha=0.1
                        )
            
            # Draw movement vectors if enabled
            if self.show_movement.get():
                movements = np.array([node['movement_influence'] for node in layer_data['nodes']])
                self.network_ax.quiver(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    movements[:, 0],
                    movements[:, 1],
                    movements[:, 2],
                    color='red',
                    alpha=0.3
                )
            
            # Recursively draw sublayers
            for sublayer in layer_data['sublayers']:
                draw_layer(sublayer, positions.mean(axis=0))
        
        draw_layer(network_state)
        self.network_ax.set_xlabel('X')
        self.network_ax.set_ylabel('Y')
        self.network_ax.set_zlabel('Z')
        self.network_canvas.draw()

    def update_movement_view(self, movement_history):
        """Update movement pattern visualization."""
        self.movement_ax.clear()
        
        # Convert movement history to numpy array
        movements = np.array(movement_history)
        
        if len(movements) > 1:
            # Plot movement trajectory
            x = np.cumsum(movements[:, 0])
            y = np.cumsum(movements[:, 1])
            
            points = np.column_stack((x, y))
            segments = np.column_stack((points[:-1], points[1:]))
            
            # Create line collection with color gradient
            from matplotlib.collections import LineCollection
            norm = plt.Normalize(0, len(segments))
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(np.arange(len(segments)))
            
            self.movement_ax.add_collection(lc)
            self.movement_ax.autoscale()
            
            # Add colorbar
            self.movement_fig.colorbar(lc, ax=self.movement_ax, label='Time')
        
        self.movement_ax.set_xlabel('X Movement')
        self.movement_ax.set_ylabel('Y Movement')
        self.movement_ax.set_title('Movement Trajectory')
        self.movement_canvas.draw()

    def update_state_analysis(self, data):
        """Update state analysis visualizations."""
        # Clear all axes
        for ax in self.state_axes.values():
            ax.clear()
        
        # Plot activation patterns
        if 'network_state' in data:
            activations = [
                np.mean([node['state'] for node in layer['nodes']], axis=0)
                for layer in self._flatten_layers(data['network_state'])
            ]
            if activations:
                self.state_axes['activation'].imshow(
                    activations,
                    aspect='auto',
                    cmap='viridis'
                )
                self.state_axes['activation'].set_title('Layer Activations')
        
        # Plot encoding patterns
        if 'network_state' in data:
            encodings = [
                node['position']
                for layer in self._flatten_layers(data['network_state'])
                for node in layer['nodes']
            ]
            if encodings:
                self.state_axes['encoding'].hist2d(
                    [e[0] for e in encodings],
                    [e[1] for e in encodings],
                    bins=20,
                    cmap='viridis'
                )
                self.state_axes['encoding'].set_title('Position Distribution')
        
        # Plot movement patterns
        if 'movement_history' in data:
            movements = np.array(data['movement_history'])
            if len(movements) > 0:
                self.state_axes['movement'].plot(
                    movements[:, 0],
                    label='X'
                )
                self.state_axes['movement'].plot(
                    movements[:, 1],
                    label='Y'
                )
                self.state_axes['movement'].set_title('Movement History')
                self.state_axes['movement'].legend()
        
        # Plot success rate
        if 'reward_history' in data:
            rewards = np.array(data['reward_history'])
            if len(rewards) > 0:
                self.state_axes['success'].plot(
                    rewards,
                    label='Reward'
                )
                # Add moving average
                window = min(len(rewards), 20)
                if window > 1:
                    moving_avg = np.convolve(
                        rewards,
                        np.ones(window)/window,
                        mode='valid'
                    )
                    self.state_axes['success'].plot(
                        moving_avg,
                        'r--',
                        label='Moving Avg'
                    )
                self.state_axes['success'].set_title('Reward History')
                self.state_axes['success'].legend()
        
        self.state_canvas.draw()

    def get_layer_color(self, depth):
        """Generate color based on layer depth."""
        import colorsys
        hue = (depth % 8) / 8.0
        return colorsys.hsv_to_rgb(hue, 0.8, 0.8)

    def _flatten_layers(self, layer_data):
        """Recursively flatten layer hierarchy."""
        layers = [layer_data]
        for sublayer in layer_data['sublayers']:
            layers.extend(self._flatten_layers(sublayer))
        return layers

    def get_all_layers(self):
        """Get list of all layer IDs."""
        def collect_layers(layer):
            layers = [layer.layer_id]
            for sublayer in layer.sub_layers:
                layers.extend(collect_layers(sublayer))
            return layers
        
        return collect_layers(self.network)

    def on_layer_select(self, event):
        """Handle layer selection change."""
        self.update_visualization()

    def on_close(self):
        """Handle window closing."""
        self.window.destroy()


class ConfigWindow:
    """Configuration window for adjusting system parameters."""
    def __init__(self, parent, config: SystemConfig, system: HiveMindSystem):
        self.parent = parent
        self.config = config
        self.system = system
        self.window = tk.Toplevel(parent)
        self.window.title("Hive Mind Configuration")
        self.window.geometry("500x600")
        self.window.resizable(False, False)
        self.window.grab_set()
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Network settings tab
        network_frame = ttk.Frame(notebook)
        notebook.add(network_frame, text="Network")
        
        # Create network settings
        settings = [
            ("Initial Nodes:", "initial_nodes", 1, 100),
            ("Initial Sub-layers:", "initial_sub_layers", 1, 10),
            ("Max Sub-layers:", "max_sub_layers", 1, 10),
            ("Encoding Dimension:", "encoding_dimension", 8, 128)
        ]
        
        for i, (label, attr, min_val, max_val) in enumerate(settings):
            ttk.Label(network_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            var = tk.IntVar(value=getattr(self.config, attr))
            spinbox = ttk.Spinbox(
                network_frame,
                from_=min_val,
                to=max_val,
                textvariable=var,
                width=10
            )
            spinbox.grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"{attr}_var", var)

        # Behavior settings tab
        behavior_frame = ttk.Frame(notebook)
        notebook.add(behavior_frame, text="Behavior")
        
        # Create behavior settings
        behavior_settings = [
            ("Growth Rate:", "growth_rate", 0.0, 1.0),
            ("Pruning Threshold:", "pruning_threshold", 0.0, 1.0),
            ("Learning Rate:", "learning_rate", 0.0, 1.0),
            ("Adaptation Rate:", "adaptation_rate", 0.0, 1.0),
            ("Movement Speed:", "movement_speed", 0.1, 10.0),
            ("Rotation Speed:", "rotation_speed", 0.1, 10.0)
        ]
        
        for i, (label, attr, min_val, max_val) in enumerate(behavior_settings):
            ttk.Label(behavior_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            var = tk.DoubleVar(value=getattr(self.config, attr))
            spinbox = ttk.Spinbox(
                behavior_frame,
                from_=min_val,
                to=max_val,
                increment=0.1,
                textvariable=var,
                width=10
            )
            spinbox.grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"{attr}_var", var)

        # Vision settings tab
        vision_frame = ttk.Frame(notebook)
        notebook.add(vision_frame, text="Vision")
        
        # Camera selection
        ttk.Label(vision_frame, text="Camera:").grid(row=0, column=0, padx=5, pady=5)
        self.camera_var = tk.IntVar(value=self.config.camera_index)
        camera_combo = ttk.Combobox(
            vision_frame,
            textvariable=self.camera_var,
            values=list(range(5)),  # Test first 5 camera indices
            state='readonly',
            width=5
        )
        camera_combo.grid(row=0, column=1, padx=5, pady=5)

        # Vision cone settings
        vision_settings = [
            ("Vision Cone Length:", "vision_cone_length", 50, 300),
            ("Vision Cone Angle:", "vision_cone_angle", 0.1, 3.14)
        ]
        
        for i, (label, attr, min_val, max_val) in enumerate(vision_settings, start=1):
            ttk.Label(vision_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            var = tk.DoubleVar(value=getattr(self.config, attr))
            spinbox = ttk.Spinbox(
                vision_frame,
                from_=min_val,
                to=max_val,
                increment=0.1,
                textvariable=var,
                width=10
            )
            spinbox.grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"{attr}_var", var)

        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            button_frame,
            text="Apply",
            command=self.apply_settings
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Save",
            command=self.save_settings
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Load",
            command=self.load_settings
        ).pack(side=tk.LEFT, padx=5)

    def apply_settings(self):
        """Apply configuration changes."""
        try:
            # Update network settings
            self.config.initial_nodes = self.initial_nodes_var.get()
            self.config.initial_sub_layers = self.initial_sub_layers_var.get()
            self.config.max_sub_layers = self.max_sub_layers_var.get()
            self.config.encoding_dimension = self.encoding_dimension_var.get()
            
            # Update behavior settings
            self.config.growth_rate = self.growth_rate_var.get()
            self.config.pruning_threshold = self.pruning_threshold_var.get()
            self.config.learning_rate = self.learning_rate_var.get()
            self.config.adaptation_rate = self.adaptation_rate_var.get()
            self.config.movement_speed = self.movement_speed_var.get()
            self.config.rotation_speed = self.rotation_speed_var.get()
            
            # Update vision settings
            new_camera_index = self.camera_var.get()
            if new_camera_index != self.config.camera_index:
                self.config.camera_index = new_camera_index
                self.system.sensory_processor = SensoryProcessor(self.config)
                
            self.config.vision_cone_length = self.vision_cone_length_var.get()
            self.config.vision_cone_angle = self.vision_cone_angle_var.get()
            
            messagebox.showinfo("Success", "Settings applied successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")

    def save_settings(self):
        """Save configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Configuration"
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=4)
                messagebox.showinfo("Success", "Settings saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_settings(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Load Configuration"
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    settings = json.load(f)
                self.config.update_from_dict(settings)
                self.update_widgets()
                messagebox.showinfo("Success", "Settings loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

    def update_widgets(self):
        """Update widget values from current configuration."""
        try:
            # Update network settings
            self.initial_nodes_var.set(self.config.initial_nodes)
            self.initial_sub_layers_var.set(self.config.initial_sub_layers)
            self.max_sub_layers_var.set(self.config.max_sub_layers)
            self.encoding_dimension_var.set(self.config.encoding_dimension)
            
            # Update behavior settings
            self.growth_rate_var.set(self.config.growth_rate)
            self.pruning_threshold_var.set(self.config.pruning_threshold)
            self.learning_rate_var.set(self.config.learning_rate)
            self.adaptation_rate_var.set(self.config.adaptation_rate)
            self.movement_speed_var.set(self.config.movement_speed)
            self.rotation_speed_var.set(self.config.rotation_speed)
            
            # Update vision settings
            self.camera_var.set(self.config.camera_index)
            self.vision_cone_length_var.set(self.config.vision_cone_length)
            self.vision_cone_angle_var.set(self.config.vision_cone_angle)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update widgets: {str(e)}")

class MainApplication:
    """Main application window and system controller."""
    def __init__(self, root):
        self.root = root
        self.root.title("Hive Mind System")
        self.root.geometry("1200x800")
        
        # Initialize queues
        self.gui_queue = queue.Queue(maxsize=50)
        self.vis_queue = queue.Queue(maxsize=50)
        
        # Initialize system
        self.config = SystemConfig()
        self.system = HiveMindSystem(self.gui_queue, self.vis_queue, self.config)
        self.visualizer = None
        
        self.create_widgets()
        self.setup_menu()
        
        # Start GUI update loop
        self.update_gui()
        
        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create main window widgets."""
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        self.start_button = ttk.Button(
            control_frame,
            text="Start",
            command=self.start_system
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop",
            command=self.stop_system,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Configure",
            command=self.show_config
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Visualize",
            command=self.show_visualizer
        ).pack(side=tk.LEFT, padx=5)
        
        # Main display canvas
        self.canvas = tk.Canvas(
            self.root,
            bg='black',
            width=self.config.display_width,
            height=self.config.display_height
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save System", command=self.save_system)
        file_menu.add_command(label="Load System", command=self.load_system)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Visualizer", command=self.show_visualizer)
        view_menu.add_command(label="Configuration", command=self.show_config)

    def update_gui(self):
        """Update GUI with latest system state."""
        try:
            while not self.gui_queue.empty():
                data = self.gui_queue.get_nowait()
                
                if 'frame' in data:
                    # Update video frame
                    frame = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame,
                        (self.canvas.winfo_width(), self.canvas.winfo_height())
                    )
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas._photo = photo
                    
                    # Draw vision cone
                    if 'position' in data:
                        self.draw_vision_cone(data['position'])
                        
                    # Draw movement vector
                    if 'movement' in data:
                        self.draw_movement_vector(data['position'], data['movement'])
                
        except Exception as e:
            logging.error(f"Error updating GUI: {e}")
            
        self.root.after(33, self.update_gui)  # ~30 FPS

    def draw_vision_cone(self, position):
        """Draw agent's vision cone."""
        x, y, rotation = position
        
        # Calculate vision cone points
        angle = self.config.vision_cone_angle
        length = self.config.vision_cone_length
        
        p1 = (x, y)
        p2 = (
            x + length * np.cos(rotation - angle),
            y + length * np.sin(rotation - angle)
        )
        p3 = (
            x + length * np.cos(rotation + angle),
            y + length * np.sin(rotation + angle)
        )
        
        # Draw vision cone
        self.canvas.create_polygon(
            p1[0], p1[1],
            p2[0], p2[1],
            p3[0], p3[1],
            fill='#00ff00',
            stipple='gray50',
            outline='#00ff00',
            width=2
        )

    def draw_movement_vector(self, position, movement):
        """Draw agent's movement vector."""
        x, y, rotation = position
        dx, dy, _ = movement
        
        # Scale movement vector
        scale = 50.0  # Visual scale factor
        end_x = x + dx * scale
        end_y = y + dy * scale
        
        # Draw movement vector
        self.canvas.create_line(
            x, y,
            end_x, end_y,
            fill='yellow',
            width=2,
            arrow=tk.LAST
        )
        
        # Draw agent position
        radius = 10
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill='#00ff00',
            outline='white',
            width=2
        )

    def start_system(self):
        """Start the hive mind system."""
        try:
            self.system.start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            logging.info("System started")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {str(e)}")
            logging.error(f"Error starting system: {e}")

    def stop_system(self):
        """Stop the hive mind system."""
        try:
            self.system.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            logging.info("System stopped")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop system: {str(e)}")
            logging.error(f"Error stopping system: {e}")

    def show_config(self):
        """Show configuration window."""
        try:
            ConfigWindow(self.root, self.config, self.system)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open configuration window: {str(e)}")
            logging.error(f"Error opening configuration window: {e}")

    def show_visualizer(self):
        """Show network visualizer window."""
        try:
            if self.visualizer is None or not tk.Toplevel.winfo_exists(self.visualizer.window):
                self.visualizer = HiveMindVisualizer(
                    self.root,
                    self.vis_queue,
                    self.system.network
                )
            else:
                self.visualizer.window.lift()  # Bring to front if already open
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open visualizer: {str(e)}")
            logging.error(f"Error opening visualizer: {e}")

    def save_system(self):
        """Save current system state."""
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".hms",
                filetypes=[("Hive Mind System", "*.hms")],
                title="Save System State"
            )
            if filepath:
                self.system.save_system(filepath)
                logging.info(f"System saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save system: {str(e)}")
            logging.error(f"Error saving system: {e}")

    def load_system(self):
        """Load system state."""
        try:
            filepath = filedialog.askopenfilename(
                defaultextension=".hms",
                filetypes=[("Hive Mind System", "*.hms")],
                title="Load System State"
            )
            if filepath:
                was_running = self.system.running
                if was_running:
                    self.stop_system()
                    
                self.system.load_system(filepath)
                
                if was_running:
                    self.start_system()
                logging.info(f"System loaded from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load system: {str(e)}")
            logging.error(f"Error loading system: {e}")

    def on_close(self):
        """Handle application closure."""
        try:
            if self.system.running:
                self.stop_system()
            
            if self.visualizer and tk.Toplevel.winfo_exists(self.visualizer.window):
                self.visualizer.window.destroy()
                
            self.root.destroy()
            logging.info("Application closed")
        except Exception as e:
            logging.error(f"Error during application closure: {e}")
            self.root.destroy()

def setup_exception_logging():
    """Setup system-wide exception logging."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logging.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Show error message to user
        messagebox.showerror(
            "Error",
            f"An unexpected error occurred:\n{str(exc_value)}\n\nCheck the log file for details."
        )
    
    # Set up the exception handler
    sys.excepthook = handle_exception

def main():
    """Main application entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("hivemind_system.log"),
            logging.StreamHandler()
        ]
    )
    
    # Setup exception handling
    setup_exception_logging()
    
    try:
        # Create main window
        root = tk.Tk()
        root.title("Hive Mind System")
        
        # Set style
        style = ttk.Style()
        style.theme_use('default')
        
        # Create application
        app = MainApplication(root)
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
