"""Utils module for functions used in the figures notebooks."""

from pathlib import Path
from typing import Union

import requests
import xarray as xr

ROOT_PATH = Path(__file__).resolve().parent.parent
CUR_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_PATH / "data"


def _relative_to_root(path: Path):
    return path.relative_to(ROOT_PATH)

def _read_dataset(filepath: Union[Path, str], fillna: bool = True):
    """
    Reads a NetCDF dataset using xarray, with an option to fill missing values.

    Parameters:
        filepath (Union[Path, str]): Path to the NetCDF file.
        fillna (bool): Whether to fill NaN values with zeros. Default is True.

    Returns:
        xarray.Dataset: The loaded dataset, with missing values optionally filled.
    """
    ds = xr.open_dataset(str(filepath))
    if fillna:
        return ds.fillna(0)
    return ds


def _download_dataset(url, filepath: Path):
    """
    Downloads a file from a given URL and saves it to the specified filepath.

    Parameters:
        url (str): URL to download the file from.
        filepath (Path): Local path to save the downloaded file.

    Raises:
        requests.RequestException: If the request to download the file fails.
    """
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise requests.RequestException(
            f"Failed to download file. Status code: {response.status_code}"
        )


def read_or_download_dataset(filename: str, url: str, fillna: bool = True):
    """
    Loads a dataset from a local file, or downloads it if not found.

    This function attempts to read a dataset from the default data directory.
    If the file is not found there, it checks the current working directory.
    If still not found, it downloads the dataset from the given URL and
    stores it in the default data directory. Optionally fills missing values.

    Parameters:
        filename (str): Name of the dataset file.
        url (str): URL to download the dataset from if it does not exist locally.
        fillna (bool, optional): If True (default), replaces missing values with zero.

    Returns:
        xarray.Dataset: The dataset loaded from file, with optional filling of NaNs.

    Raises:
        OSError: If the file cannot be written to the filesystem.
        requests.RequestException: If there is a network-related error during download.
    """
    if not DATA_PATH.is_dir():
        DATA_PATH.mkdir(parents=False, exist_ok=False)
    if (DATA_PATH / filename).is_file():
        print(f"Reading: {_relative_to_root(DATA_PATH / filename)}")
        return _read_dataset(DATA_PATH / filename, fillna)
    if (CUR_DIR / filename).is_file():
        print(f"Reading: {filename}")
        return _read_dataset(filename, fillna)
    print("Downloading data...")
    print(f"            URL: {url}")
    print(f"    Destination: {_relative_to_root(DATA_PATH / filename)}")
    _download_dataset(url, DATA_PATH / filename)
    return _read_dataset(DATA_PATH / filename, fillna)
