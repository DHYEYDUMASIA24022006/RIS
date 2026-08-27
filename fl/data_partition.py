"""Data partitioning for Federated Learning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

def iid_partition(dataset: Dataset, num_clients: int) -> dict[int, list[int]]:
    """Uniformly random split of dataset indices across clients."""
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = list(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

def dirichlet_noniid_partition(dataset: Dataset, num_clients: int, alpha: float, num_classes: int = 10) -> dict[int, list[int]]:
    """Non-IID partition using Dirichlet(alpha) distribution.
    For each class c:
        proportions ~ Dir(alpha)
        assign samples of class c to clients according to proportions
    Returns dict mapping client_id -> list of sample indices.
    """
    min_size = 0
    min_require_size = 10
    K = num_classes
    N = len(dataset)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
    net_dataidx_map = {}
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Balance based on current size
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    return net_dataidx_map

def get_client_dataloaders(dataset: Dataset, partition_map: dict[int, list[int]],
                           batch_size: int,
                           max_samples_per_client: int | None = None) -> dict[int, DataLoader]:
    """Create DataLoaders, optionally capping each client for a CPU smoke test.

    A cap is never applied unless explicitly configured. It lets the ``quick``
    runner exercise the real federated path without silently training on an
    entire image dataset on every simulated client.
    """
    dataloaders = {}
    for client_id, indices in partition_map.items():
        if max_samples_per_client is not None:
            indices = indices[:max(1, int(max_samples_per_client))]
        subset = Subset(dataset, indices)
        dataloaders[client_id] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return dataloaders

def load_dataset(name: str) -> tuple[Dataset, Dataset]:
    """Load train and test datasets (MNIST or CIFAR-10) with appropriate transforms."""
    if name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif name.lower() == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train_dataset, test_dataset
