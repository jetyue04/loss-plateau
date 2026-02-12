import random
import numpy as np
import torch

def generate_task_data(task, p, seed=42):
    """
    Generate equations for a specific task.
    Returns list of (x, y, result, task_token_name)
    """
    random.seed(seed)
    np.random.seed(seed)
    equations = []
    
    # Task token names
    task_map = {
        'div': '<DIV>',
        'add': '<ADD>',
        'sub': '<SUB>'
    }
    token_name = task_map[task]

    for x in range(p):
        # For division, y cannot be 0
        start_y = 1 if task == 'div' else 0
        for y in range(start_y, p):
            
            if task == 'div':
                # x / y = result => result = x * y^(-1)
                result = (x * pow(y, p-2, p)) % p
            elif task == 'add':
                result = (x + y) % p
            elif task == 'sub':
                result = (x - y) % p
            
            equations.append((x, y, result, token_name))
            
    return equations

def generate_multitask_dataset(tasks=['div'], p=97, train_fraction=0.5, seed=42):
    """
    Generate dataset containing one or more tasks.
    Ensures each task has the same number of examples.
    
    Args:
        tasks: List of task strings ['div', 'add', 'sub']
    """
    all_train = []
    all_val = []
    
    # We use different seeds for splitting to ensure randomness, 
    # but consistent data generation
    rng = random.Random(seed)
    
    for task in tasks:
        # Generate all possible equations for this task
        eqs = generate_task_data(task, p, seed)
        
        # Shuffle and split specific to this task
        # This ensures we have N training examples for Div, N for Add, etc.
        rng.shuffle(eqs)
        split_idx = int(len(eqs) * train_fraction)
        
        all_train.extend(eqs[:split_idx])
        all_val.extend(eqs[split_idx:])
    
    # Shuffle the combined training set so tasks are mixed in batches
    rng.shuffle(all_train)
    
    # For validation, we return a dictionary separated by task
    # This allows us to track "Div Accuracy" vs "Add Accuracy" separately
    val_data_by_task = {t: [] for t in tasks}
    for eq in all_val:
        # eq is (x, y, result, token_name)
        # map token name back to task key for sorting
        if eq[3] == '<DIV>': val_data_by_task['div'].append(eq)
        elif eq[3] == '<ADD>': val_data_by_task['add'].append(eq)
        elif eq[3] == '<SUB>': val_data_by_task['sub'].append(eq)
        
    return all_train, val_data_by_task

def create_vocab(p=97):
    """
    Create vocabulary mapping including special task tokens.
    """
    vocab = {
        'op': 0,
        'eq': 1,
        '<DIV>': 2,
        '<ADD>': 3,
        '<SUB>': 4
    }
    
    # Numbers start after special tokens
    start_idx = len(vocab)
    for i in range(p):
        vocab[f'num_{i}'] = i + start_idx
    
    return vocab

def tokenize_equation(equation, vocab):
    """
    Tokenize a single equation with task label.
    Equation: (x, y, result, task_token_str)
    Seq: [TASK, x, op, y, eq, result]
    """
    x, y, res, task_token = equation
    return [
        vocab[task_token],
        vocab[f'num_{x}'],
        vocab['op'],
        vocab[f'num_{y}'],
        vocab['eq'],
        vocab[f'num_{res}']
    ]

def prepare_data(equations, vocab):
    """
    Prepare data for training.
    """
    tokenized = [tokenize_equation(eq, vocab) for eq in equations]
    
    # X is input (TASK, a, op, b, eq)
    # y is target (c)
    # Note: tokenized length is 6. X is first 5, y is last 1.
    
    X = torch.tensor([seq[:-1] for seq in tokenized])
    y = torch.tensor([seq[-1] for seq in tokenized])
    
    return X, y