import pandas as pd
from sklearn.model_selection import train_test_split

def data_splitting(dev_dataset: pd.DataFrame, prod_dataset: pd.DataFrame) -> tuple:
    """
    Splits the development dataset into training and testing sets, maintaining class distribution.

    Args:
        dev_dataset (pd.DataFrame): The raw development dataset.
        prod_dataset (pd.DataFrame): The raw production dataset.

    Returns:
        tuple: A tuple containing:
            - train_set (pd.DataFrame): Training dataset.
            - test_set (pd.DataFrame): Testing dataset.
            - prod_dataset (pd.DataFrame): Production dataset (unchanged).
    """
    target_column = 'shot_made_flag'
    
    dev_train_set, dev_test_set = train_test_split(
        dev_dataset,
        test_size=0.2,
        stratify=dev_dataset[target_column],
        random_state=8
    )
    
    prod_train_set, prod_test_set = train_test_split(
        prod_dataset,
        test_size=0.2,
        stratify=prod_dataset[target_column],
        random_state=8
    )
    
    return dev_train_set, dev_test_set, prod_train_set, prod_test_set
    
    
