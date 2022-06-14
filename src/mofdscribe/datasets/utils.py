# -*- coding: utf-8 -*-
from pymatgen.core import train_test_split


def train_valid_test_split(x, y, train_size: float, valid_size: float, stratify):
    test_size = 1 - train_size - valid_size
