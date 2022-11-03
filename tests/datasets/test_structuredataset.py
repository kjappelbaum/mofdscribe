import pandas as pd

from mofdscribe.datasets.structuredataset import StructureDataset


def test_structuredataset(dataset_files, dataset_folder):
    # make a dataset only from the files
    structures, frame = dataset_files
    ds = StructureDataset(structures)
    assert len(ds) == len(structures)
    hashes = ds.get_decorated_graph_hashes([0, 1, 2, 3])
    assert len(hashes) == 4
    densities = ds.get_densities([0, 1, 2, 3])
    assert len(densities) == 4
