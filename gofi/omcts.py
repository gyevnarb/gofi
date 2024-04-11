from typing import Dict
from copy import deepcopy

import igp2 as ip
import logging
from igp2 import Tree

from gofi.ogoals_probabilities import OGoalsProbabilities
from gofi.occluded_factor import OccludedFactor
from gofi.orollout import ORollout

logger = logging.getLogger(__name__)


class OMCTS(ip.MCTS):
    """ An MCTS search algorithm that takes occlusions into account. """

    def _rollout(self, k: int, agent_id: int, goal: ip.Goal, tree: Tree,
                 simulator: ORollout, debug: bool, predictions: Dict[int, OGoalsProbabilities]):
        """ Run a single rollout of the MCTS search with occluded factors and store results. """
        occluded_factor = None

        # 3. Sample occluded factor instantiation
        if occluded_factor is None:
            for aid, agent_predictions in predictions.items():
                occluded_factor = agent_predictions.sample_occluded_factor()[0]
                simulator.set_occluded_factor(occluded_factor)
                break
        logger.debug(f"Occluded factor: {occluded_factor.present_elements}")

        # 4-6. Sample goal and trajectory
        samples = {}
        for aid, agent_predictions in predictions.items():
            if aid == simulator.ego_id:
                continue

            goal = agent_predictions.sample_goals_given_factor(occluded_factor)[0]
            trajectory, plan = agent_predictions.optimal_trajectory_to_goal_with_factor(goal, occluded_factor)
            simulator.update_trajectory(aid, trajectory, plan)
            samples[aid] = (goal, trajectory, occluded_factor)
            logger.debug(f"Agent {aid} sample: {plan}")

        tree.set_samples(samples)
        final_key = self._run_simulation(agent_id, goal, tree, simulator, debug)

        if self.store_results == "all":
            logger.debug(f"Storing MCTS search results for iteration {k}.")
            mcts_result = ip.MCTSResult(deepcopy(tree), samples, final_key)
            self.results.add_data(mcts_result)
