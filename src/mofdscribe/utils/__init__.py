# -*- coding: utf-8 -*-
from shutil import which


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable.
    https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script"""

    return which(name) is not None
