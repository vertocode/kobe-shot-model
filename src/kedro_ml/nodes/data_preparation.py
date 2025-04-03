import pandas as pd
from pandas import DataFrame


def data_preparation_node(dataset: pd.DataFrame) -> DataFrame:
    return dataset.dropna()