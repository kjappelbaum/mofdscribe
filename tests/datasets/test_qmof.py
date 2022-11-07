# -*- coding: utf-8 -*-
"""Test QMOF dataset."""
import numpy as np
import pytest
from pymatgen.core import IStructure

from mofdscribe.datasets.qmof_dataset import QMOFDataset


@pytest.mark.xdist_group(name="qmof-ds")
@pytest.mark.parametrize("flavor", ["all", "csd", "gcmc", "csd-gcmc"])
def test_qmof(flavor):
    """Ensure we can instantiate the QMOF dataset and access a few key methods."""
    qmof = QMOFDataset(flavor=flavor, drop_nan=False)
    assert isinstance(list(qmof.get_structures([1]))[0], IStructure)
    assert isinstance(qmof.get_labels([1]), np.ndarray)
    assert isinstance(qmof.get_years([1]), np.ndarray)
    assert len(qmof.get_years([1])) == 1
    assert len(qmof.get_years([1, 2, 4])) == 3
    assert isinstance(qmof.get_labels([1], ["outputs.pbe.bandgap"]), np.ndarray)

    # make sure we can get a subset of the dataset
    qmof_subset = qmof.get_subset([1, 2, 3, 8])
    assert isinstance(qmof_subset, QMOFDataset)
    assert len(qmof_subset._df) == 4
    assert (
        qmof._df.iloc[[1, 2, 3, 8]]["info.qmof_id"].values == qmof_subset._df["info.qmof_id"].values
    ).all()
    assert list(qmof.get_structures([1]))[0] == list(qmof_subset.get_structures([0]))[0]

    if flavor == "all":
        assert len(qmof) > 10_000
