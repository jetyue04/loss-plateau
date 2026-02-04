import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """
    Simple Transformer model for sequence prediction.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.0):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(5, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            logits: Output logits of shape (batch_size, vocab_size)
        """
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask)
        
        x = x[:, -1, :]
        logits = self.output_layer(x)
        
        return logits