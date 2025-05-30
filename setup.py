#!/usr/bin/env python

from setuptools import setup, find_packages

version = "0.3"

setup(
    name="one_pass_paper",
    version=version,
    description="One Pass Algorithms repo for paper figures",
    author="Katherine Grayson",
    author_email="katherine.grayson@bsc.es",
    url="https://github.com/kat-grayson/one_pass_algorithms_paper",
    python_requires=">=3.9",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "xarray",
        "pandas",
        "cython",
        "netcdf4",
        "cytoolz",
        "tqdm",
        "crick>=0.0.5",
        "Cartopy==0.24.*",
        "jupyter==1.1.*",
        "jupyterlab==4.4.*",
        "matplotlib==3.10.*",
        "one_pass @ git+https://github.com/DestinE-Climate-DT/one_pass@v0.8.0",
    ],
)
