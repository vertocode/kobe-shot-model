import pandas as pd

def data_feature_preparation(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant features from the dataset.

    Args:
        dataset (pd.DataFrame): Preprocessed dataset.

    Returns:
        pd.DataFrame: dataset with selected features.
    """
    selected_columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    
    return dataset[selected_columns].copy()
