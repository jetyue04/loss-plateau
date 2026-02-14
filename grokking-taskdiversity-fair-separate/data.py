import random
import numpy as np
import torch

def generate_task_data(task, p, seed=42):
    """
    Generate equations for a specific task.
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
                # x / y = result => result = x * y^(-1)
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
    all_train = []
    all_val = []
    
    rng = random.Random(seed)
    
    for task in tasks:
        eqs = generate_task_data(task, p, seed)
        rng.shuffle(eqs)
        split_idx = int(len(eqs) * train_fraction)
        
        all_train.extend(eqs[:split_idx])
        all_val.extend(eqs[split_idx:])
    
    rng.shuffle(all_train)
    
    # Separate validation data by task
    val_data_by_task = {t: [] for t in tasks}
    for eq in all_val:
        # Map tokens back to task names
        if eq[3] == '<DIV>': val_data_by_task['div'].append(eq)
        elif eq[3] == '<ADD>': val_data_by_task['add'].append(eq)
        elif eq[3] == '<SUB>': val_data_by_task['sub'].append(eq)
        elif eq[3] == '<MULT>': val_data_by_task['mult'].append(eq) # <--- NEW
        
    return all_train, val_data_by_task

def create_vocab(p=97):
    vocab = {
        'op': 0,
        'eq': 1,
        '<DIV>': 2,
        '<ADD>': 3,
        '<SUB>': 4,
        '<MULT>': 5  # <--- NEW
    }
    
    start_idx = len(vocab)
    for i in range(p):
        vocab[f'num_{i}'] = i + start_idx
    
    return vocab

def tokenize_equation(equation, vocab):
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
    tokenized = [tokenize_equation(eq, vocab) for eq in equations]
    X = torch.tensor([seq[:-1] for seq in tokenized])
    y = torch.tensor([seq[-1] for seq in tokenized])
    return X, y