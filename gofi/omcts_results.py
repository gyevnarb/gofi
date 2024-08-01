from typing import List

import igp2 as ip
from gofi.occluded_factor import OccludedFactor
from gofi.recognition.ogoals_probabilities import OGoalsProbabilities


class OMCTSResult(ip.MCTSResult):
    def __init__(self,
                 tree: "Tree" = None,
                 samples: dict = None,
                 trace: tuple = None,
                 occluded_factor: OccludedFactor = None):
        super().__init__(tree, samples, trace)
        self.__occluded_factor = occluded_factor

    @property
    def occluded_factor(self) -> OccludedFactor:
        """ The occluded factor used in the rollout. """
        return self.__occluded_factor


class AllOMCTSResults(ip.AllMCTSResult):

    @property
    def occluded_factors(self) -> List[OccludedFactor]:
        """ The list of all occluded factors associated with the MCTS planning results."""
        prediction = list(self.predictions.values())[0]
        if isinstance(prediction, OGoalsProbabilities):
            return prediction.occluded_factors
