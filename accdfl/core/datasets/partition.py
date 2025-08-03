import numpy as np


def split_dataset_uniform(dataset, n_users):
    # Shuffle the dataset to ensure randomness
    shuffled_dataset = dataset.shuffle(seed=42)

    # Calculate the number of samples per user
    num_samples = len(shuffled_dataset) // n_users

    # Create a dictionary to hold the split datasets
    split_datasets = []

    for i in range(n_users):
        start_idx = i * num_samples
        end_idx = start_idx + num_samples if i < n_users - 1 else len(shuffled_dataset)

        # Create a subset for the current user
        user_subset = shuffled_dataset.select(range(start_idx, end_idx))
        split_datasets.append(user_subset)

    return split_datasets


def split_dataset_dirichlet(dataset, n_users, alpha):
    # Get the number of classes
    labels = dataset['label']
    num_classes = len(set(labels))
    
    # Initialize a list to hold indices for each user
    user_indices = [[] for _ in range(n_users)]
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Generate the Dirichlet distribution for each class
    for cls in range(num_classes):
        # Get indices for all samples of this class
        cls_indices = np.where(np.array(labels) == cls)[0]
        
        # Get the number of samples for this class
        np.random.shuffle(cls_indices)
        num_samples = len(cls_indices)
        
        # Split the samples according to the Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * n_users)
        
        # Ensure the proportions sum to 1
        proportions = proportions / proportions.sum()
        
        # Assign samples to each user based on the proportions
        split = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
        cls_indices_split = np.split(cls_indices, split)
        
        for user, indices in enumerate(cls_indices_split):
            user_indices[user].extend(indices)
    
    # Create datasets for each user
    split_datasets = []
    for indices in user_indices:
        split_datasets.append(dataset.select(indices))
    
    return split_datasets
