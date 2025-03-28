import pandas as pd

def data_preparation_node(dev_dataset: pd.DataFrame, prod_dataset: pd.DataFrame) -> tuple:
    """
    Performs initial data cleaning on the development and production datasets, including
    handling missing values and ensuring data consistency.

    Args:
        dev_dataset (pd.DataFrame): Raw development dataset.
        prod_dataset (pd.DataFrame): Raw production dataset.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - prepared_dev_dataset: Preprocessed development dataset.
            - prepared_prod_dataset: Preprocessed production dataset.
    """

    prepared_dev_dataset = dev_dataset.dropna()
    prepared_prod_dataset = prod_dataset.dropna()

    return prepared_dev_dataset, prepared_prod_dataset