import random
import numpy as np
import torch


def generate_modular_division_dataset(p=97, train_fraction=0.5, seed=42):
    """
    Generate dataset for modular division task.
    
    Args:
        p: Prime modulus
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
    
    Returns:
        train_data: List of training equations (x, y, result)
        val_data: List of validation equations (x, y, result)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    equations = []
    
    for x in range(p):
        for y in range(1, p):
            result = (x * pow(y, p-2, p)) % p
            equations.append((x, y, result))
    
    random.shuffle(equations)
    split_idx = int(len(equations) * train_fraction)
    
    train_data = equations[:split_idx]
    val_data = equations[split_idx:]
    
    return train_data, val_data


def create_vocab(p=97):
    """
    Create vocabulary mapping for tokenization.
    
    Args:
        p: Prime modulus (determines number range)
    
    Returns:
        vocab: Dictionary mapping tokens to indices
    """
    vocab = {
        'op': 0,
        'eq': 1,
    }
    
    for i in range(p):
        vocab[f'num_{i}'] = i + 2
    
    return vocab


def tokenize_equation(equation, vocab):
    """
    Tokenize a single equation.
    
    Args:
        equation: Tuple of (a, b, c) representing a/b=c
        vocab: Vocabulary dictionary
    
    Returns:
        List of token indices
    """
    a, b, c = equation
    return [
        vocab[f'num_{a}'],
        vocab['op'],
        vocab[f'num_{b}'],
        vocab['eq'],
        vocab[f'num_{c}']
    ]


def prepare_data(equations, vocab):
    """
    Prepare data for training.
    
    Args:
        equations: List of equation tuples
        vocab: Vocabulary dictionary
    
    Returns:
        X: Input tensor (all tokens except last)
        y: Target tensor (last token)
    """
    tokenized = [tokenize_equation(eq, vocab) for eq in equations]
    
    X = torch.tensor([seq[:-1] for seq in tokenized])
    y = torch.tensor([seq[-1] for seq in tokenized])
    
    return X, y