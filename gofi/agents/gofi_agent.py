import logging
import random

import igp2 as ip
from typing import List, Dict, Tuple

from igp2 import Observation
from shapely import Point
from shapely.ops import split
import numpy as np

from gofi.recognition.ogoal_recognition import OGoalRecognition
from gofi.occluded_factor import OccludedFactor
from gofi.recognition.ogoals_probabilities import OGoalsProbabilities
from gofi.planning.omcts import OMCTS
from gofi.planning.orollout import ORollout
from gofi.planning.otree import OTree

logger = logging.getLogger(__name__)


class GOFIAgent(ip.MCTSAgent):
    """ An MCTS-based planning agent that takes occlusions into account as well. """
    OCCLUSION_AREA_THRESHOLD = 15.

    def __init__(self, occluded_factors_prior: float = 0.1, **kwargs):
        """Initialises a new GOFIAgent.

        Keyword Args:
            belief_merging_order: The order of belief merging for inferring occluded factor probabilities.
                Options are "increasing_id" or "random" or a list of agent IDs.
        """
        super().__init__(**kwargs)

        self._belief_merging_order = kwargs.get("belief_merging_order", "increasing_id")
        self._occlusions = {}

        self._occluded_factors_prior = occluded_factors_prior
        self._forced_visible_agents = None
        self._current_occluded_factor = None

        self._goal_recognition = OGoalRecognition(
            astar=self._astar,
            smoother=self._smoother,
            scenario_map=kwargs["scenario_map"],
            cost=self._cost,
            **kwargs["goal_recognition"])

        self._omcts = OMCTS(scenario_map=kwargs["scenario_map"],
                           reward=self._reward,
                           n_simulations=kwargs.get("n_simulations", 5),
                           max_depth=kwargs.get("max_depth", 5),
                           store_results=kwargs.get("store_results", "final"),
                           trajectory_agents=kwargs.get("trajectory_agents", True),
                           rollout_type=ORollout,
                           tree_type=OTree)

    def __repr__(self):
        return f"GOFIAgent(ID={self.agent_id})"

    def set_occluded_states(self, occluded_states: Dict[int, ip.AgentState]):
        """ Store the states of agents that are occluded from this agent. """
        self._occlusions = occluded_states

    def force_visible_agents(self, agent_ids: List[int]):
        """ Force the occluded factors to be used in the goal recognition. """
        self._forced_visible_agents = agent_ids

    def next_action(self, observation: ip.Observation) -> ip.Action:
        """ Returns the next action for the agent.

        If the current macro actions has finished, then updates it.
        If no macro actions are left in the plan, or we have hit the planning time step, then calls goal recognition
        and MCTS. """
        self.update_observations(observation)

        if self._k >= self._kmax or self.current_macro is None or \
                (self.current_macro.done(observation) and self._current_macro_id == len(self._macro_actions) - 1):
            self._goals = self.get_goals(observation)
            self.update_plan(observation)
            self.update_macro_action(
                self._macro_actions[0].macro_action_type,
                self._macro_actions[0].ma_args,
                observation)
            self._k = 0

        self._update_observation_with_occlusions(observation)

        if self.current_macro.done(observation):
            self._advance_macro(observation)

        self._k += 1
        return self.current_macro.next_action(observation)

    def get_occluded_factors(self, observation: ip.Observation, agents_only: bool = False) -> List[OccludedFactor]:
        """ Get a list of all possible occluded factor instantiations.
        All occluded agents follow their current lane until its end.
        """
        elements = {}
        for aid, state in self._occlusions.items():
            new_agent = ip.TrajectoryAgent(aid, state, open_loop=True, reset_trajectory=False, fps=self._omcts.fps)

            if state.speed < ip.Stop.STOP_VELOCITY:
                new_agent.set_trajectory(None, stop_seconds=100.)
            else:
                # Get random lane sequence until a road ends or reached lane limit
                max_lanes = 10
                current_lane = observation.scenario_map.best_lane_at(state.position, state.heading)
                lane_sequence = [current_lane]
                while current_lane is not None and len(lane_sequence) < max_lanes:
                    next_lane = current_lane.link.successor
                    if next_lane:
                        lane_sequence.append(next_lane[0])
                        current_lane = next_lane[0]
                    else:
                        break

                # Get path trajectory
                current_lane_final_p = lane_sequence[0].midline.interpolate(1., normalized=True)
                path = ip.Maneuver.get_lane_path_midline(lane_sequence)
                trajectory = None
                # Find which starting segment to use for path trajectory
                for segment in split(path, Point(state.position)).geoms:
                    if segment.distance(current_lane_final_p) < 0.01:
                        trajectory = np.array(segment.coords)
                        break
                velocity = np.array([state.speed] * len(trajectory))
                new_agent.set_trajectory(ip.VelocityTrajectory(trajectory, velocity))
            elements[aid] = new_agent

        if agents_only:
            return elements
        return OccludedFactor.create_all_instantiations(list(elements.values()), self._forced_visible_agents)

    def occluded_state(self, observation: ip.Observation, time: int) -> Tuple[Dict[int, ip.AgentState], ip.VelocityTrajectory]:
        """Get the estimated occluded state of an occluded factor at time t."""
        agents = self.get_occluded_factors(observation, agents_only=True)
        states = {}
        for aid, agent in agents.items():
            if len(agent.trajectory.path) < time:
                raise ValueError(f"Length of occluded trajectory is shorter than time {time}.")
            states[aid] = ip.AgentState(
                time=time,
                position=agent.trajectory.path[time],
                velocity=agent.trajectory.velocity[time],
                acceleration=agent.trajectory.acceleration[time],
                heading=agent.trajectory.heading[time],
            )
        return states, agent.trajectory.slice(0, time)

    def update_plan(self, observation: ip.Observation):
        frame = observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        occluded_factors = self.get_occluded_factors(observation)
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        self._goal_probabilities = \
            {aid: OGoalsProbabilities(self._goals, occluded_factors, occluded_factors_priors=self._occluded_factors_prior)
             for aid in frame.keys() if aid != self.agent_id and aid not in self._occlusions}

        if self._belief_merging_order == "increasing_id":
            id_order = sorted(self._goal_probabilities)
        elif self._belief_merging_order == "random":
            id_order = random.sample(list(self._goal_probabilities), len(self._goal_probabilities))
        elif isinstance(self._belief_merging_order, list):
            id_order = self._belief_merging_order
        else:
            raise ValueError(f"Unknown belief merging order: {self._belief_merging_order}")

        if not self._goal_probabilities:
            self._macro_actions, search_tree = self._mcts.search(
                agent_id=self.agent_id,
                goal=self.goal,
                frame={0: frame[0]},
                meta=agents_metadata,
                predictions=self._goal_probabilities)
            return

        previous_agent_id = None
        for agent_id in id_order:
            # No need to recognise our own goals
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

        self._current_occluded_factor = None

        self._macro_actions, search_tree = self._omcts.search(
            agent_id=self.agent_id,
            goal=self.goal,
            frame=frame,
            meta=agents_metadata,
            predictions=self._goal_probabilities)

        occluded_factor, _ = search_tree.plan_policy.select(search_tree.root)
        if isinstance(occluded_factor, OccludedFactor):
            logger.info(f"Setting occluded factor for further control: {occluded_factor}")
            self._current_occluded_factor = occluded_factor
            self._update_observation_with_occlusions(observation)

    def _update_observation_with_occlusions(self, observation: Observation):
        """ Update in-place the observation with states of occluded TrajectoryAgents in the current occluded factor. """
        if self._current_occluded_factor is not None:
            for agent in self._current_occluded_factor.present_elements:
                if agent.agent_id in observation.frame:
                    continue
                current_t = int(observation.frame[self.agent_id].time * agent.fps / self.fps)
                agent.set_start_time(current_t)
                observation.frame[agent.agent_id] = agent.state
                logger.debug(f"Updated for occluded agent {agent.agent_id} - "
                             f"Pos: {np.round(agent.state.position, 2)} -  "
                             f"Vel: {np.round(agent.state.speed, 3)}")
        return observation
