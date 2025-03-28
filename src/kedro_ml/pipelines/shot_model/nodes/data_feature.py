import pandas as pd

def data_feature_preparation(dev_dataset: pd.DataFrame, prod_dataset: pd.DataFrame) -> tuple:
    """
    Selects relevant features from the development and production datasets.

    Args:
        dev_dataset (pd.DataFrame): Preprocessed development dataset.
        prod_dataset (pd.DataFrame): Preprocessed production dataset.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - prepared_dev_dataset: Development dataset with selected features.
            - prepared_prod_dataset: Production dataset with selected features.
    """
    selected_columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    
    prepared_dev_dataset = dev_dataset[selected_columns].copy()
    prepared_prod_dataset = prod_dataset[selected_columns].copy()
    
    return prepared_dev_dataset, prepared_prod_dataset
