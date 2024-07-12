from typing import Dict, Tuple, List

import igp2 as ip
from igp2 import Goal, AgentState, GoalsProbabilities, StateTrajectory, Agent, Observation

from gofi.occluded_factor import OccludedFactor
from gofi.map.omap import OMap


class ORollout(ip.Rollout):
    def __init__(self,
                 ego_id: int,
                 initial_frame: dict[int, ip.AgentState],
                 metadata: dict[int, ip.AgentMetadata],
                 scenario_map: OMap,
                 fps: int = 10,
                 open_loop_agents: bool = False,
                 trajectory_agents: bool = False,
                 t_max: int = 200,
                 occluded_factor: OccludedFactor = None):
        """ Initialise a new rollout with possible occluded factors.

        Args:
            ego_id: The id of the ego vehicle.
            initial_frame: The initial frame of the rollout.
            metadata: metadata describing the agents in the environment
            scenario_map: current road layout
            fps: frame rate of simulation
            open_loop_agents: Whether non-ego agents follow open-loop control
            trajectory_agents: Whether to use predicted trajectories directly or CL macro actions for non-egos
            t_max: Maximum rollout time step length
            occluded_factor: The occluded factor to consider in the environment.
            """
        super().__init__(ego_id, initial_frame, metadata, scenario_map, fps, open_loop_agents, trajectory_agents, t_max)
        self._initial_frame_agents = list(initial_frame)
        self._init_occluded_factor = occluded_factor
        self._occluded_factor = occluded_factor
        self._hide_occluded = False

    def set_occluded_factor(self, occluded_factor: OccludedFactor):
        """ Override the current occluded factor instantiation of the environment."""
        self._occluded_factor = occluded_factor
        for element in occluded_factor.present_elements:
            self.agents[element.agent_id] = element
            self.initial_frame[element.agent_id] = element.state

    def _get_observation(self, frame: Dict[int, AgentState], agent_id: int = None) -> Observation:
        if self._hide_occluded and self._occluded_factor is not None and agent_id == self.ego_id:
            of_ids = [a.agent_id for a in self._occluded_factor.elements]
            frame = {aid: state for aid, state in frame.items() if aid not in of_ids}
        return super()._get_observation(frame, agent_id)

    def hide_occluded(self):
        """ Hide the occluded factor from the ego in simulation."""
        self._hide_occluded = True

    def reset(self):
        """ Reset the rollout to its initial state and removes occluded factors from the environment."""
        super().reset()
        self._occluded_factor = self._init_occluded_factor
        self._agents = {aid: agent for aid, agent in self.agents.items()
                        if aid in self._initial_frame_agents}
        self._initial_frame = {aid: state for aid, state in self.initial_frame.items()
                               if aid in self._initial_frame_agents}
        self._hide_occluded = False

    @property
    def occluded_factor(self) -> OccludedFactor:
        """ Return the current occluded factor in the rollout. """
        return self._occluded_factor
