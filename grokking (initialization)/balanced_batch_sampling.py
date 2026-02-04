import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random


class BalancedTaskBatchSampler(Sampler):
    """
    Sampler that creates balanced batches where each batch contains
    equal proportions of each task.
    
    For example, with 50-50 division-addition and batch_size=512:
    Each batch will have exactly 256 division and 256 addition examples.
    """
    
    def __init__(self, task_labels, task_mix, batch_size, shuffle=True):
        """
        Args:
            task_labels: List of task labels for each example (e.g., ['division', 'addition', ...])
            task_mix: Dict mapping task names to proportions (e.g., {'division': 0.5, 'addition': 0.5})
            batch_size: Total batch size
            shuffle: Whether to shuffle examples within tasks
        """
        self.task_labels = task_labels
        self.task_mix = task_mix
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by task
        self.task_indices = {}
        for task_name in task_mix.keys():
            self.task_indices[task_name] = [
                i for i, label in enumerate(task_labels) if label == task_name
            ]
        
        # Calculate samples per task per batch
        self.samples_per_task = {}
        for task_name, proportion in task_mix.items():
            self.samples_per_task[task_name] = int(batch_size * proportion)
        
        # Adjust last task to make total exactly batch_size (handles rounding)
        total = sum(self.samples_per_task.values())
        if total != batch_size:
            last_task = list(task_mix.keys())[-1]
            self.samples_per_task[last_task] += (batch_size - total)
        
        # Calculate number of batches (limited by smallest task)
        min_batches = min(
            len(indices) // self.samples_per_task[task_name]
            for task_name, indices in self.task_indices.items()
        )
        self.num_batches = min_batches
    
    def __iter__(self):
        # Shuffle task indices if requested
        task_indices_shuffled = {}
        for task_name, indices in self.task_indices.items():
            if self.shuffle:
                indices_copy = indices.copy()
                random.shuffle(indices_copy)
                task_indices_shuffled[task_name] = indices_copy
            else:
                task_indices_shuffled[task_name] = indices
        
        # Create balanced batches
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            for task_name in self.task_mix.keys():
                start_idx = batch_idx * self.samples_per_task[task_name]
                end_idx = start_idx + self.samples_per_task[task_name]
                batch_indices.extend(task_indices_shuffled[task_name][start_idx:end_idx])
            
            # Shuffle within batch
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class MultiTaskDataset(Dataset):
    """
    Dataset that stores both data and task labels.
    """
    
    def __init__(self, X, y, task_labels):
        """
        Args:
            X: Input tensor
            y: Target tensor
            task_labels: List of task names for each example
        """
        self.X = X
        self.y = y
        self.task_labels = task_labels
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_balanced_dataloaders(train_data, val_data, vocab, task_mix, batch_size, 
                                multi_task=True):
    """
    Create dataloaders with balanced task sampling in each batch.
    
    Args:
        train_data: List of training examples (task, x, y, result) or (x, y, result)
        val_data: List of validation examples
        vocab: Vocabulary dictionary
        task_mix: Dict of task proportions (e.g., {'division': 0.5, 'addition': 0.5})
        batch_size: Batch size
        multi_task: Whether using multi-task data format
    
    Returns:
        train_loader: DataLoader with balanced batches
        val_loader: DataLoader with balanced batches
    """
    from data import prepare_data
    
    # Prepare data
    X_train, y_train = prepare_data(train_data, vocab, multi_task=multi_task)
    X_val, y_val = prepare_data(val_data, vocab, multi_task=multi_task)
    
    if multi_task:
        # Extract task labels
        train_task_labels = [example[0] for example in train_data]
        val_task_labels = [example[0] for example in val_data]
        
        # Create datasets
        train_dataset = MultiTaskDataset(X_train, y_train, train_task_labels)
        val_dataset = MultiTaskDataset(X_val, y_val, val_task_labels)
        
        # Create balanced batch samplers
        train_sampler = BalancedTaskBatchSampler(
            train_task_labels, task_mix, batch_size, shuffle=True
        )
        val_sampler = BalancedTaskBatchSampler(
            val_task_labels, task_mix, batch_size, shuffle=False
        )
        
        # Create dataloaders with batch samplers
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_sampler=val_sampler
        )
        
        print(f"\nBalanced batch sampling enabled:")
        print(f"  Batch size: {batch_size}")
        print(f"  Each batch contains:")
        for task_name, proportion in task_mix.items():
            count = train_sampler.samples_per_task[task_name]
            print(f"    - {count} {task_name} examples ({proportion*100:.0f}% of batch)")
        print(f"  Total batches per epoch: {len(train_loader)}")
        print(f"\n    Training examples per task per epoch:")
        for task_name in task_mix.keys():
            task_count = sum(1 for label in train_task_labels if label == task_name)
            examples_per_epoch = min(task_count, len(train_loader) * train_sampler.samples_per_task[task_name])
            print(f"    - {task_name}: {examples_per_epoch} examples")
        print()
        
    else:
        # Standard dataloaders for single-task
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader