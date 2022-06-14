# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Union

import pandas as pd


def length_check(df: pd.DataFrame, expected_length: int) -> None:
    if not len(df) == expected_length:
        raise ValueError("Length of dataframe does not match expected length.")


def check_all_file_exists(filelist: List[Union[str, Path]]) -> None:
    for f in filelist:
        if not os.path.exists(f):
            raise ValueError(f"File {f} does not exist.")
