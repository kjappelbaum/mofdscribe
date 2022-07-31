# -*- coding: utf-8 -*-
"""Utils for MOFDescribe."""

import sys
from shutil import which

import numpy as np

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


def nan_array(size):
    return np.full(size, np.nan)


def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable.

    Taken from https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    Args:
        name (str): executable name

    Returns:
        bool: True if the executable in PATH
    """
    return which(name) is not None


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
