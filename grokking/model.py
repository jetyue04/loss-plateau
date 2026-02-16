'''
model.py defines the transformer architecture used for grokking experiments.

This module implements a simple decoder-only transformer that learns to predict
the result of modular arithmetic operations. The model uses causal masking to
prevent attending to future tokens and outputs a distribution over the vocabulary
for the next token prediction.
'''

import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """
    A simple decoder-only transformer model for sequence-to-token prediction.
    
    This model learns to predict the result of modular arithmetic operations given
    an input sequence of the form: <TASK> x op y eq. It uses causal self-attention
    and predicts a single output token (the result).
    
    Architecture:
        - Token embedding layer
        - Learned positional encoding
        - Stacked transformer encoder layers with causal masking
        - Linear output layer projecting to vocabulary size
    """
    
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.0):
        """
        Initialize the transformer model.
        
        :param vocab_size: Size of the vocabulary (number of unique tokens)
        :param d_model: Dimension of model embeddings and hidden states
        :param nhead: Number of attention heads in multi-head attention
        :param num_layers: Number of transformer encoder layers to stack
        :param dropout: Dropout probability applied in transformer layers
        """
        super().__init__()
        
        # Token embeddings: map each token ID to a d_model dimensional vector
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encodings: learned embeddings for each position in sequence
        # Fixed to 5 positions since input sequences are always length 5
        self.pos_encoding = nn.Embedding(5, d_model)
        
        # Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Standard 4x expansion in FFN
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: map final hidden state to vocabulary logits
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Processes an input sequence through embeddings, positional encoding,
        transformer layers with causal masking, and projects the final hidden
        state to vocabulary logits.
        
        :param x: Input tensor of token indices, shape (batch_size, seq_len)
        :return: Logits over vocabulary, shape (batch_size, vocab_size)
        """
        # Create position indices for positional encoding
        # Shape: (1, seq_len) -> broadcasts to (batch_size, seq_len)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        
        # Add token embeddings and positional encodings
        # Shape: (batch_size, seq_len, d_model)
        x = self.embedding(x) + self.pos_encoding(positions)
        
        # Create causal attention mask (upper triangular = True means masked)
        # This prevents positions from attending to future positions
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Apply transformer layers with causal mask
        # Shape: (batch_size, seq_len, d_model)
        x = self.transformer(x, mask=mask)
        
        # Extract final position's hidden state for prediction
        # Shape: (batch_size, d_model)
        x = x[:, -1, :]
        
        # Project to vocabulary logits
        # Shape: (batch_size, vocab_size)
        logits = self.output_layer(x)
        
        return logits
