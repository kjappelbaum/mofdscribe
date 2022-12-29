from mofdscribe.featurizers.bu.racs import ModularityCommunityCenteredRACS
from mofdscribe.mof import MOF


def test_modularitycommunity_racs(cof_1_structure):
    featurizer = ModularityCommunityCenteredRACS()
    feats = featurizer.featurize(MOF(cof_1_structure))
    labels = featurizer.feature_labels()
    assert len(feats) == len(labels)

    # cof preset
    featurizer = ModularityCommunityCenteredRACS.from_preset("cof")
    feats = featurizer.featurize(MOF(cof_1_structure))
    labels = featurizer.feature_labels()
    assert len(feats) == len(labels)
