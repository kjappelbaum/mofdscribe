# -*- coding: utf-8 -*-
from mofdscribe.featurizers.base import MOFBaseFeaturizer
from mofdscribe.featurizers.utils.extend import operates_on_structuregraph

# operates on needs to be extended to graphs


# cannot use here the MOFBaseFeaturizer because the input is not always a structure
# not super sure about this yet, but I think, over the long run, I'd like this to only accept graphs
# i'd like to have some kind of transformer objects that handle the conversion from structure to graph
# if needed
# all of this could then be orchestrated in a pipeline object
@operates_on_structuregraph
class GraphFeaturizer(MOFBaseFeaturizer):
    pass
