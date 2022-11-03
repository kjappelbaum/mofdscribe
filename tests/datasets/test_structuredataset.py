from mofdscribe.datasets.structuredataset import StructureDataset
import pandas as pd 

def test_structuredataset(dataset_files, dataset_folder):
    # make a dataset only from the files 
    structures, frame = dataset_files
    ds = StructureDataset(structures)
    assert len(ds) == len(structures)
    hashes = ds.get_decorated_graph_hashes([0,1,2,3])
    assert len(hashes) == 4