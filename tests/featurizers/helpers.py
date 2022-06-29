# -*- coding: utf-8 -*-
"""Helper functions for tests."""
import json


def is_jsonable(x):
    """Test if a object is JSONable."""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
