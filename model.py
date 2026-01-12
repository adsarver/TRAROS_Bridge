# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisionEncoder2d(nn.Module):
    def __init__(self, image_size=64):
        super(VisionEncoder2d, self).__init__()
                
        # Input shape: (agents, 1, image_size, image_size)
        # Based off of TinyLidarNet from: https://arxiv.org/pdf/2410.07447
        # Adapted for flexible input sizes with adaptive pooling
        # Added dropout for cross-track generalization
        
        # Calculate adaptive kernel sizes based on image size
        # For image_size=64: k1=5, s1=2 -> 30x30
        # For image_size=128: k1=7, s1=2 -> 61x61
        k1 = max(3, min(7, image_size // 12))
        k2 = max(3, min(5, image_size // 16))
        
        self.conv_layers = nn.Sequential(
            # First conv: Reduce spatial dimensions by ~2x
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=k1, stride=2, padding=k1//2),
            nn.GroupNorm(1, 24),
            nn.ReLU(),
            
            # Second conv: Reduce by ~2x
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=k2, stride=2, padding=k2//2),
            nn.GroupNorm(1, 36),
            nn.ReLU(),
            
            # Third conv: Reduce by ~2x
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 48),
            nn.ReLU(),
            
            # Fourth conv: Increase channels, same spatial size
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            
            # Fifth conv: Final feature extraction
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            
            # Adaptive pooling to fixed size for consistent output
            # This ensures output is always (64, 4, 4) regardless of input size
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # Fixed output size: 64 channels * 4 * 4 = 1024
        self.output_size = 64 * 4 * 4

    def forward(self, scan_tensor):
        return self.conv_layers(scan_tensor)
    
class VisionEncoder(nn.Module):
    def __init__(self, num_scan_beams=1080):
        super(VisionEncoder, self).__init__()
        
        # Input shape: (batch_size, 1, num_scan_beams)
        # Based off of TinyLidarNet from: https://arxiv.org/pdf/2410.07447
        # Added dropout for cross-track generalization
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4),
            nn.GroupNorm(1, 24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4),
            nn.GroupNorm(1, 36),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2),
            nn.GroupNorm(1, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten()
        )
        
        # Calculate the output size of the conv layers
        dummy_input = torch.randn(1, 1, num_scan_beams)
        self.output_size = self._get_conv_output_size(dummy_input)

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, scan_tensor):
        # Handle batched inputs from replay buffer: [batch, agents, 1, scan] -> [batch*agents, 1, scan]
        original_shape = scan_tensor.shape
        if len(original_shape) == 4:  # [batch, agents, 1, scan]
            batch_size, num_agents = original_shape[0], original_shape[1]
            scan_tensor = scan_tensor.reshape(batch_size * num_agents, original_shape[2], original_shape[3])
            features = self.conv_layers(scan_tensor)
            # Reshape back: [batch*agents, features] -> [batch, agents, features]
            features = features.reshape(batch_size, num_agents, -1)
            return features
        else:
            return self.conv_layers(scan_tensor)


class RecurrentActorNetwork(nn.Module):
    """
    LSTM-enhanced policy network with downsampled temporal memory.
    Maintains a buffer of observations sampled every N steps for efficient long-term memory.
    
    Args:
        memory_length: Number of historical observations to keep (e.g., 5)
        memory_stride: Sample every Nth observation (e.g., 5 = keep every 5th step)
        Total temporal window = memory_length * memory_stride steps
        Example: 5 memories at stride 5 = 25 steps = 0.5 seconds at 50Hz
    """
    def __init__(
        self, 
        state_dim=3, 
        action_dim=2,
        encoder=None,
        lstm_hidden_size=128,
        lstm_num_layers=1,
        memory_length=5,  # Keep 5 observations
        memory_stride=5   # Sample every 5 steps, total window of 25 steps
        ):
        super(RecurrentActorNetwork, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length  # Number of observations in sequence
        self.memory_stride = memory_stride  # Steps between samples
        self.step_counter = 0  # Track when to sample
        
        # Combine CNN features with state vector
        feature_input_size = conv_output_size + state_dim
        
        # Project features to LSTM input size with gradual compression + dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization for cross-track generalization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, lstm_hidden_size),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        # Input will be sequence of memory_length observations
        # Using nn.LSTM with flatten_parameters disabled for vmap compatibility
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Policy head (maps LSTM output to action distribution) - Adjusted for larger hidden size
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(32, action_dim)
        
        # log_std head (learnable parameter for exploration)
        self.log_std_head = nn.Linear(32, action_dim)
        
    @torch.jit.export
    def get_init_hidden(self, batch_size:int, device:torch.device, transpose:bool=False):
        """Initialize hidden and cell states for LSTM."""
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    @torch.jit.export
    def create_observation_buffer(self, batch_size:int, device:torch.device):
        """
        Create a buffer to store recent observations for LSTM input.
        Returns a buffer of shape (batch_size, memory_length, feature_size)
        """
        # Placeholder - will be filled with actual encoded features
        return torch.zeros(batch_size, self.memory_length, self.lstm_hidden_size).to(device)

    def forward(self, scan_tensor, state_tensor, obs_buffer, hidden_h, hidden_c):
        """
        Args:
            scan_tensor: (batch, 1, lidar_beams) - Current observation
            state_tensor: (batch, state_dim) - Current state
            obs_buffer: (batch, memory_length, lstm_hidden_size) - Historical features buffer
            hidden: Optional LSTM hidden state tuple (h, c)
        Returns:
            loc, scale: Action distribution parameters
            obs_buffer: Updated observation buffer (shifted + new observation)
            hidden: Updated LSTM hidden state (for next step)
        """
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device
        # Update batch_size after potential reshaping
        batch_size = scan_tensor.shape[0]
        
        # Initialize buffers if not provided
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if hidden_h is None or hidden_c is None:
            hidden = self.get_init_hidden(batch_size, device)
        else:
            # Transpose hidden from [batch, num_layers, hidden] to [num_layers, batch, hidden]
            hidden = (hidden_h.transpose(0, 1).contiguous(), hidden_c.transpose(0, 1).contiguous())
        
        # CNN feature extraction and concatenation
        vision_features = self.conv_layers(scan_tensor)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        # Project to LSTM input size
        current_feature = self.feature_projection(combined_features)
        
        # Update observation buffer (shift left, add new observation at end)
        obs_buffer_updated = torch.cat([
            obs_buffer[:, 1:, :],  # Drop oldest observation
            current_feature.unsqueeze(1)  # Add newest observation
        ], dim=1)
        
        # LSTM forward pass on observation sequence
        lstm_out, hidden_new = self.lstm(obs_buffer_updated, hidden)
        lstm_final = lstm_out[:, -1, :]
        
        # Policy head
        x = self.fc_layers(lstm_final)
        loc = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5.0, 2.0)  # Prevent scale collapse: min=exp(-5)≈0.007
        scale = torch.exp(log_std)
        scale = torch.clamp(scale, min=0.01, max=10.0)  # Hard floor on scale
        
        # Transpose hidden states from [num_layers, batch, hidden] to [batch, num_layers, hidden]
        hidden_transposed = (hidden_new[0].transpose(0, 1).contiguous(), hidden_new[1].transpose(0, 1).contiguous())
        
        return loc, scale, obs_buffer_updated, hidden_transposed[0], hidden_transposed[1]

class RecurrentCriticNetwork(nn.Module):
    """
    LSTM-enhanced value network with downsampled temporal memory.
    Maintains a buffer of observations sampled every N steps for efficient long-term memory.
    """
    def __init__(
        self, 
        state_dim=3, 
        encoder=None,
        lstm_hidden_size=128,
        lstm_num_layers=1,
        memory_length=5,  # Keep 5 observations
        memory_stride=5   # Sample every 5 steps
        ):
        super(RecurrentCriticNetwork, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        
        # Combine CNN features with state vector
        feature_input_size = conv_output_size + state_dim
        
        # Project features to LSTM input size with gradual compression + dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization for cross-track generalization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, lstm_hidden_size),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        # Using nn.LSTM with flatten_parameters disabled for vmap compatibility
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Value head (maps LSTM output to state value) - Adjusted for larger hidden size
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def get_init_hidden(self, batch_size, device, transpose=False):
        """Initialize hidden and cell states for LSTM."""
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    def create_observation_buffer(self, batch_size, device):
        """Create a buffer to store recent observations for LSTM input."""
        return torch.zeros(batch_size, self.memory_length, self.lstm_hidden_size).to(device)

    def forward(self, scan_tensor, state_tensor, obs_buffer, hidden_h, hidden_c):
        """
        Args:
            scan_tensor: (batch, 1, lidar_beams) or (T, B, 1, lidar_beams) - Current observation
            state_tensor: (batch, state_dim) or (T, B, state_dim) - Current state
            obs_buffer: (batch, memory_length, lstm_hidden_size) - Historical features buffer
            hidden: Optional LSTM hidden state tuple (h, c)
        Returns:
            value: State value estimate
            obs_buffer: Updated observation buffer
            hidden: Updated LSTM hidden state
        """
        
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device
        
        # Initialize buffers if not provided
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if hidden_h is None or hidden_c is None:
            hidden = self.get_init_hidden(batch_size, device)
        else:
            # Transpose hidden from [batch, num_layers, hidden] to [num_layers, batch, hidden]
            hidden = (hidden_h.transpose(0, 1), hidden_c.transpose(0, 1))
            
        # CNN feature extraction and concatenation
        vision_features = self.conv_layers(scan_tensor)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        # Project to LSTM input size
        current_feature = self.feature_projection(combined_features)
        # Update observation buffer (sliding window)
        obs_buffer_updated = torch.cat([
            obs_buffer[:, 1:, :],  # Drop oldest
            current_feature.unsqueeze(1)  # Add newest
        ], dim=1)
        
        
        # LSTM forward pass
        lstm_out, hidden_new = self.lstm(obs_buffer_updated, hidden)
        
        # Take last timestep output
        lstm_final = lstm_out[:, -1, :]
        
        # Value head
        value = self.fc_layers(lstm_final)
        
        # Transpose hidden states from [num_layers, batch, hidden] to [batch, num_layers, hidden]
        hidden_transposed = (hidden_new[0].transpose(0, 1).contiguous(), hidden_new[1].transpose(0, 1).contiguous())
        
        return value, obs_buffer_updated, hidden_transposed[0], hidden_transposed[1]

# Keep old ActorNetwork for backwards compatibility
class ActorNetwork(nn.Module):
    """
    A 1D CNN-based policy network (Actor).
    It takes a LIDAR scan and outputs a probability distribution
    over the continuous actions (steering and speed).
    """
    def __init__(
        self, 
        state_dim=3, 
        action_dim=2,
        encoder=None
        ):
        super(ActorNetwork, self).__init__()
        
        # Input shape: (num_agents, 1, num_scan_beams)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        fc_input_size = conv_output_size + state_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size),
            nn.GroupNorm(1, fc_input_size),
            nn.ReLU(),
            nn.Linear(fc_input_size, 100),
            nn.GroupNorm(1, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.GroupNorm(1, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.GroupNorm(1, 10),
            nn.ReLU(),
        )

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(10, action_dim)
        
        # log_std head
        self.log_std_head = nn.Linear(10, action_dim)

    def forward(self, scan_tensor, state_tensor):
        if scan_tensor.ndim == 4:
            T, B = scan_tensor.shape[0:2]
            # Flatten Time and Batch dims for Conv1D
            scan_tensor = scan_tensor.reshape(T * B, 1, -1) 
            state_tensor = state_tensor.reshape(T * B, -1)
            unflatten_output = True
        else:
            unflatten_output = False
            
        # NN Layers            
        vision_features = self.conv_layers(scan_tensor)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        x = self.fc_layers(combined_features)
        
        # Heads
        loc = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        scale = torch.exp(log_std)
        
        if unflatten_output:
            loc = loc.view(T, B, -1)
            scale = scale.view(T, B, -1)

        return loc, scale
class CriticNetwork(nn.Module):
    def __init__(self, state_dim=3, encoder=None):
        super(CriticNetwork, self).__init__()
        
        # Vision Stream (LIDAR)
        self.conv_layers = encoder
        
        conv_output_size = self.conv_layers.output_size
        fc_input_size = conv_output_size + state_dim
        
        # Combined Fully Connected Layers - DEEPER & WIDER for multi-map value estimation
        # Increased capacity to learn track-invariant value function across 18 different tracks
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 256),  # Wider first layer
            nn.GroupNorm(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Additional layer
            nn.GroupNorm(1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Gradual reduction
            nn.GroupNorm(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Additional layer
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, scan_tensor, state_tensor):
        if scan_tensor.ndim == 4:
            T, B = scan_tensor.shape[0:2]
            # Flatten Time and Batch dims for Conv1D
            scan_tensor = scan_tensor.reshape(T * B, 1, -1) 
            state_tensor = state_tensor.reshape(T * B, -1)
            unflatten_output = True
        else:
            unflatten_output = False
        
        # NN Layers            
        vision_features = self.conv_layers(scan_tensor)
        
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        value = self.fc_layers(combined_features)
        
        if unflatten_output:
            return value.view(T, B, 1)
        else:
            return value