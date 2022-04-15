from matminer.featurizers.base import BaseFeaturizer


class RACS(BaseFeaturizer):
    def __init__(self) -> None:
        ...

    def featurize(self, structure):
        ...

    def feature_labels(self):
        ...

    def citations(self):
        ...

    def implementors(self):
        ...
