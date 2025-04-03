import pandas as pd
import requests
import io

from pandas import DataFrame

devFileUrl = 'https://raw.githubusercontent.com/tciodaro/eng_ml/main/data/dataset_kobe_dev.parquet'
prodFileUrl = 'https://raw.githubusercontent.com/tciodaro/eng_ml/main/data/dataset_kobe_prod.parquet'

def data_acquisition_node(type: str) -> DataFrame | None:
    """
    Downloads Parquet datasets dynamically from remote URLs and loads them into pandas DataFrames.
    
    This function fetches both the development and production datasets from the specified URLs,
    reads them in memory, and returns them as DataFrames. If the download fails, an exception is raised.

    Returns:
        tuple: A tuple containing two pandas DataFrames (dev_data, prod_data), 
               one for each dataset.
    
    Raises:
        Exception: If either dataset cannot be downloaded due to a request error.
    """
    data = None
    if type == "dev":
        # Download dev dataset
        dev_response = requests.get(devFileUrl)
        if dev_response.status_code == 200:
            print("Development file downloaded successfully!")
            data = pd.read_parquet(io.BytesIO(dev_response.content))
        else:
            raise Exception(f"Failed to download the development file. Status code: {dev_response.status_code}")

    if type == "prod":
        # Download prod dataset
        prod_response = requests.get(prodFileUrl)
        if prod_response.status_code == 200:
            print("Production file downloaded successfully!")
            data = pd.read_parquet(io.BytesIO(prod_response.content))
        else:
            raise Exception(f"Failed to download the production file. Status code: {prod_response.status_code}")
    
    return data
