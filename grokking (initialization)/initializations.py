"""
Custom initialization methods for accelerating grokking.

Includes:
- Low-rank initialization: Initialize weights as product of two smaller matrices
- Sparse initialization: Initialize most weights to zero, keep only a fraction
- Small random initialization: Start with very small weights
"""

import torch
import torch.nn as nn
import math


def low_rank_init(weight, rank_ratio=0.5):
    """
    Initialize weight matrix as a low-rank product: W = U @ V^T
    
    Args:
        weight: Weight tensor to initialize (shape: [out_features, in_features])
        rank_ratio: Ratio of rank to min(in_features, out_features) (default: 0.5)
    
    The weight matrix W is factorized as:
        W = U @ V^T
    where U is [out_features, rank] and V is [in_features, rank]
    
    This forces the initial weight matrix to be low-rank, which may encourage
    simpler, more generalizable solutions.
    """
    out_features, in_features = weight.shape
    
    # Calculate rank
    max_rank = min(out_features, in_features)
    rank = max(1, int(max_rank * rank_ratio))
    
    # Initialize U and V with Xavier initialization
    std = math.sqrt(2.0 / (in_features + out_features))
    U = torch.randn(out_features, rank) * std
    V = torch.randn(in_features, rank) * std
    
    # Compute low-rank weight matrix
    low_rank_weight = U @ V.T
    
    # Copy to weight tensor
    with torch.no_grad():
        weight.copy_(low_rank_weight)
    
    return weight


def sparse_init(weight, sparsity=0.9, std=None):
    """
    Initialize weight matrix with most entries set to zero (sparse).
    
    Args:
        weight: Weight tensor to initialize
        sparsity: Fraction of weights to set to zero (default: 0.9 = 90% sparse)
        std: Standard deviation for non-zero weights (default: Xavier std)
    
    Only (1 - sparsity) fraction of weights are non-zero.
    This encourages the network to learn with fewer connections.
    """
    if std is None:
        # Use Xavier initialization std for non-zero weights
        fan_in = weight.size(1) if weight.dim() >= 2 else weight.size(0)
        fan_out = weight.size(0) if weight.dim() >= 2 else weight.size(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
    
    with torch.no_grad():
        # Initialize all weights
        weight.normal_(0, std)
        
        # Create sparse mask
        mask = torch.rand_like(weight) > sparsity
        
        # Apply mask (set sparsity% of weights to zero)
        weight.mul_(mask.float())
        
        # Rescale non-zero weights to maintain variance
        # (since we zeroed out many weights)
        scale = 1.0 / math.sqrt(1.0 - sparsity)
        weight.mul_(scale)
    
    return weight


def small_init(weight, scale=0.01):
    """
    Initialize weights to very small random values.
    
    Args:
        weight: Weight tensor to initialize
        scale: Scale factor for initialization (default: 0.01)
    
    Starting with very small weights may help the network
    find simpler solutions before memorizing.
    """
    with torch.no_grad():
        fan_in = weight.size(1) if weight.dim() >= 2 else weight.size(0)
        fan_out = weight.size(0) if weight.dim() >= 2 else weight.size(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        # Standard Xavier init but scaled down
        weight.normal_(0, std * scale)
    
    return weight


def apply_initialization(model, init_type='default', **kwargs):
    """
    Apply custom initialization to all linear layers in model.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization
            - 'default': Standard Xavier/Kaiming (no change)
            - 'low_rank': Low-rank initialization
            - 'sparse': Sparse initialization
            - 'small': Small random initialization
        **kwargs: Additional arguments for initialization functions
            - rank_ratio: For low_rank (default: 0.5)
            - sparsity: For sparse (default: 0.9)
            - scale: For small (default: 0.01)
    
    Returns:
        model: Model with applied initialization
    """
    if init_type == 'default':
        # Keep default PyTorch initialization
        return model
    
    init_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if init_type == 'low_rank':
                rank_ratio = kwargs.get('rank_ratio', 0.5)
                low_rank_init(module.weight, rank_ratio=rank_ratio)
                print(f"  Applied low-rank init (rank_ratio={rank_ratio}) to {name}")
                
            elif init_type == 'sparse':
                sparsity = kwargs.get('sparsity', 0.9)
                sparse_init(module.weight, sparsity=sparsity)
                print(f"  Applied sparse init (sparsity={sparsity}) to {name}")
                
            elif init_type == 'small':
                scale = kwargs.get('scale', 0.01)
                small_init(module.weight, scale=scale)
                print(f"  Applied small init (scale={scale}) to {name}")
            
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            
            # Always initialize bias to zero
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            
            init_count += 1
    
    print(f"  Total layers initialized: {init_count}")
    return model


def analyze_initialization(model, verbose=True):
    """
    Analyze the initialization of a model.
    
    Args:
        model: PyTorch model
        verbose: Print detailed statistics
    
    Returns:
        stats: Dictionary of initialization statistics
    """
    stats = {
        'layer_stats': [],
        'total_params': 0,
        'total_nonzero': 0,
    }
    
    if verbose:
        print("\n" + "="*70)
        print("INITIALIZATION ANALYSIS")
        print("="*70)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            
            # Compute statistics
            layer_stat = {
                'name': name,
                'shape': weight.shape,
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'min': weight.min().item(),
                'max': weight.max().item(),
                'num_params': weight.numel(),
                'num_nonzero': (weight != 0).sum().item(),
                'sparsity': 1.0 - (weight != 0).float().mean().item(),
            }
            
            # Compute approximate rank using SVD
            try:
                U, S, V = torch.svd(weight)
                # Effective rank: number of singular values > 1% of max
                threshold = 0.01 * S.max()
                effective_rank = (S > threshold).sum().item()
                layer_stat['effective_rank'] = effective_rank
                layer_stat['rank_ratio'] = effective_rank / min(weight.shape)
            except:
                layer_stat['effective_rank'] = None
                layer_stat['rank_ratio'] = None
            
            stats['layer_stats'].append(layer_stat)
            stats['total_params'] += layer_stat['num_params']
            stats['total_nonzero'] += layer_stat['num_nonzero']
            
            if verbose:
                print(f"\n{name}:")
                print(f"  Shape: {layer_stat['shape']}")
                print(f"  Mean: {layer_stat['mean']:.6f}, Std: {layer_stat['std']:.6f}")
                print(f"  Range: [{layer_stat['min']:.6f}, {layer_stat['max']:.6f}]")
                print(f"  Sparsity: {layer_stat['sparsity']*100:.2f}%")
                if layer_stat['effective_rank'] is not None:
                    print(f"  Effective Rank: {layer_stat['effective_rank']} "
                          f"(ratio: {layer_stat['rank_ratio']:.2f})")
    
    stats['overall_sparsity'] = 1.0 - (stats['total_nonzero'] / stats['total_params'])
    
    if verbose:
        print("\n" + "="*70)
        print(f"Overall Statistics:")
        print(f"  Total Parameters: {stats['total_params']:,}")
        print(f"  Non-zero Parameters: {stats['total_nonzero']:,}")
        print(f"  Overall Sparsity: {stats['overall_sparsity']*100:.2f}%")
        print("="*70 + "\n")
    
    return stats
