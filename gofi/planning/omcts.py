from typing import Dict, List, Tuple
from copy import deepcopy

import igp2 as ip
import logging
from igp2 import MCTSAction

from gofi.planning.otree import OTree
from gofi.recognition.ogoals_probabilities import OGoalsProbabilities
from gofi.planning.orollout import ORollout
from gofi.omcts_results import OMCTSResult, AllOMCTSResults

logger = logging.getLogger(__name__)


class OMCTS(ip.MCTS):
    """ An MCTS search algorithm that takes occlusions into account. """
    def __init__(self, *args, **kwargs):
        super(OMCTS, self).__init__(*args, **kwargs)
        self._allow_hide_occluded = kwargs.get("allow_hide_occluded", True)
        self._current_occluded_factor = None
        self._hide_occluded = False

    def _rollout(self,
                 k: int,
                 agent_id: int,
                 goal: ip.Goal,
                 tree: OTree,
                 simulator: ORollout,
                 debug: bool,
                 predictions: Dict[int, OGoalsProbabilities]):
        """ Run a single rollout of the MCTS search with occluded factors and store results. """
        occluded_factor = None

        # 3. Sample occluded factor instantiation
        for aid, agent_predictions in predictions.items():
            occluded_factor = agent_predictions.sample_occluded_factor()[0]
            simulator.set_occluded_factor(occluded_factor)

            # Set the start time of the occluded factor to account for different FPSs
            for element in occluded_factor.present_elements:
                if isinstance(element, ip.TrajectoryAgent):
                    element.set_start_time(int(tree.root.state[0].time * simulator.fps / self.env_fps))

            self._hide_occluded = tree.set_occlusions(occluded_factor, self._allow_hide_occluded)
            if self._hide_occluded:
                # If an occluded factor is present sometimes we want to pretend it is not there to
                #  test the ego for missing the occluded factor.
                simulator.hide_occluded()
            self._current_occluded_factor = occluded_factor
            break
        logger.debug(f"  Occluded factor: {occluded_factor.present_elements}")

        # 4-6. Sample goal and trajectory
        samples = {}
        for aid, agent_predictions in predictions.items():
            if aid == simulator.ego_id:
                continue

            agent_goal = agent_predictions.sample_goals_given_factor(occluded_factor)[0]
            trajectory, plan = agent_predictions.optimal_trajectory_to_goal_with_factor(agent_goal, occluded_factor)
            simulator.update_trajectory(aid, trajectory, plan)
            samples[aid] = (agent_goal, trajectory, occluded_factor)
            logger.debug(f"  Agent {aid} sample: {plan}")

        final_key = self._run_simulation(agent_id, goal, tree, simulator, debug)
        logger.debug(f"  Final key: {final_key}")

        if self.store_results == "all":
            logger.debug(f"  Storing MCTS search results for iteration {k}.")
            mcts_result = OMCTSResult(deepcopy(tree), samples, final_key, occluded_factor)
            self.results.add_data(mcts_result)

    def _reset_results(self):
        """ Resets the stored results in the OMCTS instance."""
        if self.store_results is None:
            self.results = None
        elif self.store_results == 'final':
            self.results = OMCTSResult()
        elif self.store_results == 'all':
            self.results = AllOMCTSResults()

    def _create_tree(self,
                     agent_id: int,
                     frame: Dict[int, ip.AgentState],
                     goal: ip.Goal,
                     predictions: Dict[int, OGoalsProbabilities]):
        """ Creates a new MCTS tree to store results. """
        root = self._create_node(self.to_key(None), agent_id, frame, goal)
        tree = OTree(root, occluded_factors=list(predictions.values())[0].occluded_factors)
        return tree

    def reset(self):
        """ Reset OMCTS by calling super and removing the pre-existing occluded factor."""
        super().reset()
        self._current_occluded_factor = None
        self._hide_occluded = False

    def to_key(self, plan: List[MCTSAction] = None) -> Tuple[str, ...]:
        base_key = super().to_key(plan)
        if (self._current_occluded_factor is not None and
                not self._current_occluded_factor.no_occlusions and
                not self._hide_occluded):
            return ("Super", str(self._current_occluded_factor)) + base_key[1:]
        else:
            return ("Super", ) + base_key
