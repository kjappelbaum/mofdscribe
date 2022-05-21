from matminer.featurizers.base import BaseFeaturizer


class BuildingBlockFeaturizer(BaseFeaturizer):
    def __init__(self, featurizer) -> None:
        super().__init__()

    def featurize(self, nodes, linkers):
        ...
