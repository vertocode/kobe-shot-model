import pandas as pd
from sklearn.model_selection import train_test_split

def data_splitting(dataset) -> tuple:
    """
    Splits the dataset into training and testing sets, maintaining class distribution.

    Args:
        dataset (pd.DataFrame): The raw dataset.

    Returns:
        tuple: A tuple containing:
            - train_set (pd.DataFrame): Training dataset.
            - test_set (pd.DataFrame): Testing dataset.
    """
    target_column = 'shot_made_flag'
    
    train_set, test_set = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset[target_column],
        random_state=8
    )
    
    return train_set, test_set
    
    
