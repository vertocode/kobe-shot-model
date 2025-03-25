import pandas as pd
import requests
import io

devFileUrl = 'https://raw.githubusercontent.com/tciodaro/eng_ml/main/data/dataset_kobe_dev.parquet'

def data_loader_node() -> pd.DataFrame:
    """
    Downloads a Parquet dataset dynamically from a remote URL and loads it into a pandas DataFrame.

    This function fetches the dataset from the specified `devFileUrl`, reads it in memory, and
    returns it as a DataFrame. If the download fails, an exception is raised.

    Returns:
        pd.DataFrame: A DataFrame containing the downloaded dataset.

    Raises:
        Exception: If the dataset cannot be downloaded due to a request error.
    """
    response = requests.get(devFileUrl)
    
    if response.status_code == 200:
        print("File downloaded successfully!")
        data = pd.read_parquet(io.BytesIO(response.content))
        return data
    else:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")