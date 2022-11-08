import pandas as pd

from mofdscribe.datasets.structuredataset import StructureDataset, FrameDataset


def test_structuredataset(dataset_files, dataset_folder):
    # make a dataset only from the files
    structures, frame = dataset_files
    ds = StructureDataset(structures)
    assert len(ds) == len(structures)
    hashes = ds.get_decorated_graph_hashes([0, 1, 2, 3])
    assert len(hashes) == 4
    densities = ds.get_densities([0, 1, 2, 3])
    assert len(densities) == 4

    # make a dataset from the files and a dataframe
    frame = pd.read_json(frame[0])
    ds = StructureDataset(
        structures,
        frame,
        structure_name_column="info.basename",
        decorated_graph_hash_column="info.decorated_graph_hash",
    )
    # only two of them are in the dataframe
    assert len(ds) == 2
    hashes = ds.get_decorated_graph_hashes([0, 1])
    assert len(hashes) == 2

    # make a dataset from a folder and a dataframe
    ds = StructureDataset.from_folder_and_dataframe(
        dataset_folder,
        dataframe=frame,
        structure_name_column="info.basename",
        decorated_graph_hash_column="info.decorated_graph_hash",
    )
    # only two of them are in the dataframe
    assert len(ds) == 2
    hashes = ds.get_decorated_graph_hashes([0, 1])
    assert len(hashes) == 2


def test_framedataset(dataset_files):
    _, frame = dataset_files
    frame = pd.read_json(frame[0])
    ds = FrameDataset(frame,  structure_name_column="info.basename",
        decorated_graph_hash_column="info.decorated_graph_hash")
    # only two of them are in the dataframe
    assert len(ds) == 2
    hashes = ds.get_decorated_graph_hashes([0, 1])
    assert len(hashes) == 2