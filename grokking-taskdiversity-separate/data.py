'''
data.py contains functions for generating modular arithmetic datasets, creating vocabularies,
and preparing data for training transformer models on grokking experiments.

This module supports multiple arithmetic operations (division, addition, subtraction, multiplication)
performed modulo a prime number p, and handles dataset splitting and tokenization.
'''

import random
import numpy as np
import torch


def generate_task_data(task, p, seed=42):
    """
    Generate all possible equations for a specific modular arithmetic task.
    
    For a given prime p, this generates all valid pairs (x, y) and computes the result
    of the operation modulo p. Each equation is represented as a tuple containing
    the operands, result, and task identifier token.
    
    :param task: Type of arithmetic operation ('div', 'add', 'sub', or 'mult')
    :param p: Prime modulus for modular arithmetic
    :param seed: Random seed for reproducibility
    :return: List of tuples, each containing (x, y, result, task_token)
    """
    random.seed(seed)
    np.random.seed(seed)
    equations = []
    
    # Task token names
    task_map = {
        'div': '<DIV>',
        'add': '<ADD>',
        'sub': '<SUB>',
        'mult': '<MULT>'
    }
    token_name = task_map[task]

    for x in range(p):
        # For division, y cannot be 0. For others, it can.
        start_y = 1 if task == 'div' else 0
        
        for y in range(start_y, p):
            
            if task == 'div':
                # x / y = result => result = x * y^(-1) mod p
                # Using Fermat's little theorem: y^(-1) = y^(p-2) mod p
                result = (x * pow(y, p-2, p)) % p
            elif task == 'add':
                result = (x + y) % p
            elif task == 'sub':
                result = (x - y) % p
            elif task == 'mult':
                result = (x * y) % p
            
            equations.append((x, y, result, token_name))
            
    return equations


def generate_multitask_dataset(tasks=['div'], p=97, train_fraction=0.5, seed=42):
    """
    Generate a multi-task dataset with separate training and validation splits.
    
    Creates equations for multiple arithmetic tasks, splits each task's data into
    training and validation sets, then combines training data across tasks while
    keeping validation data separated by task for per-task evaluation.
    
    :param tasks: List of task names to include (e.g., ['div', 'add', 'mult'])
    :param p: Prime modulus for modular arithmetic
    :param train_fraction: Fraction of data to use for training (rest for validation)
    :param seed: Random seed for reproducibility
    :return: Tuple of (all_train, val_data_by_task) where:
        - all_train: List of training equations from all tasks combined
        - val_data_by_task: Dictionary mapping task names to their validation equations
    """
    all_train = []
    all_val = []
    
    rng = random.Random(seed)
    
    # Generate and split data for each task
    for task in tasks:
        eqs = generate_task_data(task, p, seed)
        rng.shuffle(eqs)
        split_idx = int(len(eqs) * train_fraction)
        
        all_train.extend(eqs[:split_idx])
        all_val.extend(eqs[split_idx:])
    
    # Shuffle combined training data so tasks are mixed
    rng.shuffle(all_train)
    
    # Separate validation data by task for per-task evaluation
    val_data_by_task = {t: [] for t in tasks}
    for eq in all_val:
        # Map tokens back to task names
        if eq[3] == '<DIV>': 
            val_data_by_task['div'].append(eq)
        elif eq[3] == '<ADD>': 
            val_data_by_task['add'].append(eq)
        elif eq[3] == '<SUB>': 
            val_data_by_task['sub'].append(eq)
        elif eq[3] == '<MULT>': 
            val_data_by_task['mult'].append(eq)
        
    return all_train, val_data_by_task


def create_vocab(p=97):
    """
    Create a vocabulary mapping tokens to integer indices.
    
    The vocabulary includes:
    - Special tokens: 'op' (operator separator), 'eq' (equals sign)
    - Task tokens: '<DIV>', '<ADD>', '<SUB>', '<MULT>'
    - Number tokens: 'num_0' through 'num_{p-1}' for all numbers mod p
    
    :param p: Prime modulus, determines the range of number tokens
    :return: Dictionary mapping token strings to integer indices
    """
    vocab = {
        'op': 0,
        'eq': 1,
        '<DIV>': 2,
        '<ADD>': 3,
        '<SUB>': 4,
        '<MULT>': 5
    }
    
    # Add number tokens starting after special tokens
    start_idx = len(vocab)
    for i in range(p):
        vocab[f'num_{i}'] = i + start_idx
    
    return vocab


def tokenize_equation(equation, vocab):
    """
    Convert an equation tuple into a sequence of token indices.
    
    The equation format is: <TASK> x op y eq result
    Where <TASK> is one of <DIV>, <ADD>, <SUB>, <MULT>
    
    :param equation: Tuple of (x, y, result, task_token)
    :param vocab: Vocabulary dictionary mapping tokens to indices
    :return: List of token indices representing the equation
    """
    x, y, res, task_token = equation
    return [
        vocab[task_token],      # Task identifier
        vocab[f'num_{x}'],      # First operand
        vocab['op'],            # Operator separator
        vocab[f'num_{y}'],      # Second operand
        vocab['eq'],            # Equals sign
        vocab[f'num_{res}']     # Result
    ]


def prepare_data(equations, vocab):
    """
    Tokenize equations and prepare PyTorch tensors for training.
    
    Converts a list of equations into input sequences (X) and target labels (y).
    The input contains all tokens except the final result, and the target is the result token.
    
    :param equations: List of equation tuples to tokenize
    :param vocab: Vocabulary dictionary for token conversion
    :return: Tuple of (X, y) where:
        - X: Tensor of shape (num_equations, 5) containing input sequences
        - y: Tensor of shape (num_equations,) containing target result tokens
    """
    tokenized = [tokenize_equation(eq, vocab) for eq in equations]
    X = torch.tensor([seq[:-1] for seq in tokenized])  # All tokens except result
    y = torch.tensor([seq[-1] for seq in tokenized])   # Only the result token
    return X, y
