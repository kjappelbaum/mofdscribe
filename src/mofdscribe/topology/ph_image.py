from typing import List, Tuple

from matminer.featurizers.base import BaseFeaturizer


class PHImage(BaseFeaturizer):
    def __init__(self, atom_types=Tuple[str]) -> None:
        ...

    def feature_labels(self) -> List[str]:
        return ...

    def citations(self):
        return []

    def implementors(self):
        return ["Kevin Maik Jablonka"]
