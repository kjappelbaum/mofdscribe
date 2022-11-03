# -*- coding: utf-8 -*-
"""Sanity checks for datasets."""
import os
from typing import List, Union

import pandas as pd

__all__ = ["length_check", "check_all_file_exists"]


def length_check(df: pd.DataFrame, expected_length: int) -> None:
    """Ensure the length of the dataframe is as expected."""
    if not len(df) == expected_length:
        raise ValueError(
            "Length of dataframe does not match expected length. Found {}, expected {}".format(
                len(df), expected_length
            )
        )


def check_all_file_exists(filelist: List[Union[str, os.PathLike]]) -> None:
    """Ensure that all expected structures have been downloaded."""
    for f in filelist:
        if not os.path.exists(f):
            raise ValueError(f"File {f} does not exist.")
