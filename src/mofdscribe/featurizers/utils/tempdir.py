# -*- coding: utf-8 -*-
"""Setting the directory for temporary files."""
import os

TEMPDIR = os.getenv("MOFDSCRIBE_TEMPDIR", os.getcwd())
