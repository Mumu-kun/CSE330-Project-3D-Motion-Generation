"""
Model architectures for Human Motion Animation Generation.

This module contains:
- AutoregressiveContextEncoder: Encodes motion context sequentially
- FlowMatchingNetwork: Generates motion sequences using flow matching

Compatible with MoMask input format: dim-263 feature vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class AutoregressiveContextEncoder(nn.Module):
    """
    Autoregressive Context Encoder for motion generation using GRU.
    
    Processes motion sequences sequentially to encode contextual information.
    Input: dim-263 feature vectors (HumanML3D format)
    Output: Context embeddings for flow matching
    """
    
    def __init__(
        self,
        input_dim: int = 263,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_length: int = 196,
        bidirectional: bool = False
    ):
        """
        Initialize Autoregressive Context Encoder with GRU.
        
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Hidden dimension for GRU layers
            num_layers: Number of GRU layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Text encoding (placeholder - will be integrated with CLIP or similar)
        self.text_encoder = None  # TODO: Initialize text encoder
        
        # Output projection
        # If bidirectional, GRU output is 2 * hidden_dim
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Linear(gru_output_dim, hidden_dim)
        
    @property
    def output_dim(self) -> int:
        """Output dimension of the context encoder."""
        return self.hidden_dim
    
    def forward(
        self,
        motion: torch.Tensor,
        text: Optional[List[str]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GRU-based context encoder.
        
        Args:
            motion: Input motion features (batch, seq_len, input_dim)
            text: Optional text descriptions (list of strings)
            mask: Optional padding mask (batch, seq_len) - True for valid positions
            
        Returns:
            context: Encoded context (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = motion.shape
        
        # Project input
        x = self.input_projection(motion)  # (batch, seq_len, hidden_dim)
        
        # TODO: Integrate text encoding if provided
        if text is not None:
            # text_features = self.text_encoder(text)  # TODO: Implement
            # Option 1: Add text features to first timestep
            # Option 2: Concatenate text features to each timestep
            # Option 3: Use text features to initialize hidden state
            pass
        
        # Handle padding mask for GRU
        if mask is not None:
            # Convert mask: True for valid positions, False for padding
            # pack_padded_sequence expects lengths
            lengths = mask.sum(dim=1).cpu()  # (batch,)
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output_packed, hidden = self.gru(x_packed)
            # Unpack the sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True, total_length=seq_len
            )
        else:
            # No mask provided, process full sequence
            output, hidden = self.gru(x)  # output: (batch, seq_len, hidden_dim or 2*hidden_dim)
        
        # Output projection
        context = self.output_projection(output)  # (batch, seq_len, hidden_dim)
        
        return context


class FlowMatchingNetwork(nn.Module):
    """
    Flow Matching Network for motion generation.
    
    Generates motion sequences using continuous normalizing flows (flow matching).
    Takes context from AutoregressiveContextEncoder and generates motion.
    """
    
    def __init__(
        self,
        context_dim: int = 512,
        motion_dim: int = 263,
        hidden_dim: int = 512,
        num_layers: int = 12,
        dropout: float = 0.1,
        num_timesteps: int = 1000
    ):
        """
        Initialize Flow Matching Network.
        
        Args:
            context_dim: Dimension of context from encoder
            motion_dim: Dimension of motion features (263 for HumanML3D)
            hidden_dim: Hidden dimension for network layers
            num_layers: Number of flow matching layers
            dropout: Dropout probability
            num_timesteps: Number of timesteps for flow matching
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context projection
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        
        # Flow matching layers
        layers = []
        for i in range(num_layers):
            layers.append(
                FlowMatchingLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    motion_dim=motion_dim if i == num_layers - 1 else hidden_dim,
                    dropout=dropout
                )
            )
        self.flow_layers = nn.ModuleList(layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, motion_dim)
        
    def forward(
        self,
        context: torch.Tensor,
        motion: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through flow matching network.
        
        Args:
            context: Context from encoder (batch, seq_len, context_dim)
            motion: Optional motion for training (batch, seq_len, motion_dim)
            timestep: Optional timestep for flow matching (batch,)
            
        Returns:
            output: Generated or predicted motion (batch, seq_len, motion_dim)
        """
        batch_size, seq_len, _ = context.shape
        
        # Project context
        x = self.context_projection(context)  # (batch, seq_len, hidden_dim)
        
        # Time embedding
        if timestep is None:
            # During inference, use random timesteps
            timestep = torch.rand(batch_size, device=context.device)
        
        t_emb = self.time_embedding(timestep.unsqueeze(-1))  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)
        
        # Combine context and time
        x = x + t_emb
        
        # Apply flow matching layers
        for layer in self.flow_layers:
            x = layer(x, motion if motion is not None else None)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class FlowMatchingLayer(nn.Module):
    """
    Single layer for flow matching network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        motion_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, motion_dim)
        )
        
    def forward(self, x: torch.Tensor, motion: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through flow matching layer.
        
        Args:
            x: Input features (batch, seq_len, input_dim)
            motion: Optional motion for residual connection (batch, seq_len, motion_dim)
            
        Returns:
            output: Processed features (batch, seq_len, motion_dim)
        """
        output = self.layer(x)
        
        # Residual connection if motion is provided
        if motion is not None and motion.shape[-1] == output.shape[-1]:
            output = output + motion
        
        return output
