import logging
import random

import igp2 as ip
from typing import List

import numpy as np

from gofi.ogoal_recognition import OGoalRecognition
from gofi.occluded_factor import OccludedFactor
from gofi.ogoals_probabilities import OGoalsProbabilities
from gofi.omcts import OMCTS
from gofi.orollout import ORollout

logger = logging.getLogger(__name__)


class GOFIAgent(ip.MCTSAgent):
    """ An MCTS-based planning agent that takes occlusions into account as well. """
    OCCLUSION_AREA_THRESHOLD = 15.

    def __init__(self, **kwargs):
        """Initialises a new GOFIAgent.

        Keyword Args:
            belief_merging_order: The order of belief merging for inferring occluded factor probabilities.
                Options are "increasing_id" or "random" or a list of agent IDs.
        """
        super().__init__(**kwargs)

        self._belief_merging_order = kwargs.get("belief_merging_order", "increasing_id")
        self._occlusions = {}

        self._goal_recognition = OGoalRecognition(
            astar=self._astar,
            smoother=self._smoother,
            scenario_map=kwargs["scenario_map"],
            cost=self._cost,
            **kwargs["goal_recognition"])

        self._mcts = OMCTS(scenario_map=kwargs["scenario_map"],
                           reward=self._reward,
                           n_simulations=kwargs.get("n_simulations", 5),
                           max_depth=kwargs.get("max_depth", 5),
                           store_results=kwargs.get("store_results", "final"),
                           trajectory_agents=kwargs.get("trajectory_agents", True),
                           rollout_type=ORollout)

    def __repr__(self):
        return f"GOFIAgent(ID={self.agent_id})"

    def set_occluded_states(self, occluded_states: dict[int, ip.AgentState]):
        """ Store the states of agents that are occluded from this agent. """
        self._occlusions = occluded_states

    def get_occluded_factors(self, observation: ip.Observation) -> List[OccludedFactor]:
        """ Get a list of all possible occluded factor instantiations. """
        elements = []
        for aid, state in self._occlusions.items():
            distance = 100.  # Distance to cover from the start position in a straight line
            duration = 10.  # Duration under which to cover the distance
            direction = state.velocity / np.linalg.norm(state.velocity)
            path = np.array([state.position, state.position + distance * direction])
            velocity = np.array([distance / duration, distance / duration])
            new_agent = ip.TrajectoryAgent(aid, state, open_loop=True, reset_trajectory=False)
            new_agent.set_trajectory(ip.VelocityTrajectory(path, velocity))
            elements.append(new_agent)

        return OccludedFactor.create_all_instantiations(elements)

    def update_plan(self, observation: ip.Observation):
        frame = observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        occluded_factors = self.get_occluded_factors(observation)
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        self._goal_probabilities = {aid: OGoalsProbabilities(self._goals, occluded_factors)
                                    for aid in frame.keys() if aid != self.agent_id}

        if self._belief_merging_order == "increasing_id":
            id_order = self._belief_merging_order = sorted(self._goal_probabilities)
        elif self._belief_merging_order == "random":
            id_order = random.sample(list(self._goal_probabilities), len(self._goal_probabilities))
        elif isinstance(self._belief_merging_order, list):
            id_order = self._belief_merging_order
        else:
            raise ValueError(f"Unknown belief merging order: {self._belief_merging_order}")

        previous_agent_id = None
        for agent_id in id_order:
            # No need to recongise our own goals
            if agent_id == self.agent_id:
                continue

            # Perform belief merging by using previous agent posterior as next agent prior
            if previous_agent_id is not None:
                for factor, pz in self._goal_probabilities[previous_agent_id].occluded_factors_probabilities.items():
                    self._goal_probabilities[agent_id].occluded_factors_priors[factor] = pz

            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=self._goal_probabilities[agent_id],
                observed_trajectory=self._observations[agent_id][0],
                agent_id=agent_id,
                frame_ini=self._observations[agent_id][1],
                frame=frame,
                visible_region=visible_region)
            previous_agent_id = agent_id

        # Set merged beliefs to be the same for all agents, i.e., the beliefs of the last agent in the merging order
        pz = self._goal_probabilities[previous_agent_id].occluded_factors_probabilities
        for agent_id, probabilities in self._goal_probabilities.items():
            probabilities.set_merged_occluded_factors_probabilities(pz)

        self._macro_actions = self._mcts.search(
            agent_id=self.agent_id,
            goal=self.goal,
            frame=frame,
            meta=agents_metadata,
            predictions=self._goal_probabilities)
