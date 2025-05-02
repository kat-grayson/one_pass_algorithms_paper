"""Utils module for functions used in the figures notebooks."""

import os
import requests
import xarray as xr

def read_or_download_dataset(filename: str, url: str, fillna: bool = True):
    """
    Reads a dataset from a local file or downloads it from a URL if not present, with optional filling of missing values.

    This function checks whether a file exists locally. If it does not,
    it downloads the file from the specified URL and saves it locally.
    Once the file is available, it opens the file as an xarray dataset.
    If `fillna` is True, missing (NaN) values in the dataset are replaced with zero.

    Parameters:
        filename (str): The name of the local file to read or save the download to.
        url (str): The URL to download the file from if it does not exist locally.
        fillna (bool, optional): If True (default), fill missing values in the dataset with zero.

    Returns:
        xarray.Dataset: The dataset loaded from the specified file, optionally with missing values filled.

    Raises:
        OSError: If the file cannot be written locally.
        requests.RequestException: If there is a network-related error during download.
    """
    if not os.path.isfile(filename):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    ds = xr.open_dataset(filename)
    if fillna:
        return ds.fillna(0)
    return ds
