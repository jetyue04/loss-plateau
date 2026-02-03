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


def generate_modular_addition_dataset(p=97, train_fraction=0.5, seed=42):
    """
    Generate dataset for modular addition task.
    
    Args:
        p: Prime modulus
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
    
    Returns:
        train_data: List of training equations (x, y, result)
        val_data: List of validation equations (x, y, result)
    """
    random.seed(seed + 1)  # Different seed to ensure different split
    np.random.seed(seed + 1)
    
    equations = []
    
    for x in range(p):
        for y in range(p):
            result = (x + y) % p
            equations.append((x, y, result))
    
    random.shuffle(equations)
    split_idx = int(len(equations) * train_fraction)
    
    train_data = equations[:split_idx]
    val_data = equations[split_idx:]
    
    return train_data, val_data


def generate_modular_subtraction_dataset(p=97, train_fraction=0.5, seed=42):
    """
    Generate dataset for modular subtraction task.
    
    Args:
        p: Prime modulus
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
    
    Returns:
        train_data: List of training equations (x, y, result)
        val_data: List of validation equations (x, y, result)
    """
    random.seed(seed + 2)  # Different seed to ensure different split
    np.random.seed(seed + 2)
    
    equations = []
    
    for x in range(p):
        for y in range(p):
            result = (x - y) % p
            equations.append((x, y, result))
    
    random.shuffle(equations)
    split_idx = int(len(equations) * train_fraction)
    
    train_data = equations[:split_idx]
    val_data = equations[split_idx:]
    
    return train_data, val_data


def generate_mixed_task_dataset(p=97, train_fraction=0.5, seed=42, 
                                task_mix={'division': 0.5, 'addition': 0.5},
                                fair_comparison=True):
    """
    Generate mixed dataset with multiple modular arithmetic tasks.
    
    Args:
        p: Prime modulus
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
        task_mix: Dictionary mapping task names to their proportions
                  e.g., {'division': 0.5, 'addition': 0.5}
        fair_comparison: If True, use same number of training examples per task
                        regardless of task_mix proportions. This ensures fair
                        comparison across different task mixtures.
    
    Returns:
        train_data: List of training tuples (task, x, y, result)
        val_data: List of validation tuples (task, x, y, result)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Validate task_mix
    if abs(sum(task_mix.values()) - 1.0) > 1e-6:
        raise ValueError("Task proportions must sum to 1.0")
    
    # Generate datasets for each task
    task_datasets = {}
    
    if 'division' in task_mix:
        div_train, div_val = generate_modular_division_dataset(p, train_fraction, seed)
        task_datasets['division'] = (div_train, div_val)
    
    if 'addition' in task_mix:
        add_train, add_val = generate_modular_addition_dataset(p, train_fraction, seed)
        task_datasets['addition'] = (add_train, add_val)
    
    if 'subtraction' in task_mix:
        sub_train, sub_val = generate_modular_subtraction_dataset(p, train_fraction, seed)
        task_datasets['subtraction'] = (sub_train, sub_val)
    
    if fair_comparison:
        # FAIR COMPARISON MODE:
        # Use ALL training examples from each task, regardless of task_mix
        # The task_mix only determines batch proportions, not dataset size
        # This ensures each task sees the same amount of data across different runs
        
        train_data = []
        val_data = []
        
        for task_name, (task_train, task_val) in task_datasets.items():
            # Add task identifier to each equation
            train_with_task = [(task_name, x, y, result) for x, y, result in task_train]
            val_with_task = [(task_name, x, y, result) for x, y, result in task_val]
            
            train_data.extend(train_with_task)
            val_data.extend(val_with_task)
        
        # Shuffle mixed data
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        print(f"\n  Fair Comparison Mode: Using ALL training examples per task")
        print(f"   Task mix ({task_mix}) determines BATCH proportions only")
        print(f"   Each task uses its full training set:")
        for task_name, (task_train, task_val) in task_datasets.items():
            print(f"     - {task_name}: {len(task_train)} train, {len(task_val)} val")
        print(f"   Total: {len(train_data)} train, {len(val_data)} val examples")
        
    else:
        # ORIGINAL MODE (UNFAIR):
        # Scale dataset size by task_mix proportions
        # This means different task mixes see different amounts of data per task
        
        train_data = []
        val_data = []
        
        for task_name, (task_train, task_val) in task_datasets.items():
            proportion = task_mix[task_name]
            
            # Add task identifier to each equation
            train_with_task = [(task_name, x, y, result) for x, y, result in task_train]
            val_with_task = [(task_name, x, y, result) for x, y, result in task_val]
            
            train_data.extend(train_with_task)
            val_data.extend(val_with_task)
        
        # Shuffle mixed data
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        print(f"\n   Original Mode (Unfair): Dataset size varies with task_mix")
    
    return train_data, val_data


def create_vocab(p=97, multi_task=False):
    """
    Create vocabulary mapping for tokenization.
    
    Args:
        p: Prime modulus (determines number range)
        multi_task: If True, include separate operation tokens for different tasks
    
    Returns:
        vocab: Dictionary mapping tokens to indices
    """
    if multi_task:
        vocab = {
            'op_div': 0,
            'op_add': 1,
            'op_sub': 2,
            'eq': 3,
        }
        offset = 4
    else:
        vocab = {
            'op': 0,
            'eq': 1,
        }
        offset = 2
    
    for i in range(p):
        vocab[f'num_{i}'] = i + offset
    
    return vocab


def tokenize_equation(equation, vocab, multi_task=False):
    """
    Tokenize a single equation.
    
    Args:
        equation: Tuple of (a, b, c) for single task, or (task, a, b, c) for multi-task
        vocab: Vocabulary dictionary
        multi_task: Whether this is a multi-task equation
    
    Returns:
        List of token indices
    """
    if multi_task:
        task, a, b, c = equation
        
        # Map task to operation token
        task_to_op = {
            'division': 'op_div',
            'addition': 'op_add',
            'subtraction': 'op_sub'
        }
        op_token = vocab[task_to_op[task]]
        
        return [
            vocab[f'num_{a}'],
            op_token,
            vocab[f'num_{b}'],
            vocab['eq'],
            vocab[f'num_{c}']
        ]
    else:
        a, b, c = equation
        return [
            vocab[f'num_{a}'],
            vocab['op'],
            vocab[f'num_{b}'],
            vocab['eq'],
            vocab[f'num_{c}']
        ]


def prepare_data(equations, vocab, multi_task=False):
    """
    Prepare data for training.
    
    Args:
        equations: List of equation tuples
        vocab: Vocabulary dictionary
        multi_task: Whether equations include task identifiers
    
    Returns:
        X: Input tensor (all tokens except last)
        y: Target tensor (last token)
    """
    tokenized = [tokenize_equation(eq, vocab, multi_task) for eq in equations]
    
    X = torch.tensor([seq[:-1] for seq in tokenized])
    y = torch.tensor([seq[-1] for seq in tokenized])
    
    return X, y