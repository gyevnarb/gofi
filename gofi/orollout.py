from typing import Dict

import igp2 as ip
from igp2 import Goal, AgentState, GoalsProbabilities

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
                 t_max: int = 1000,
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
        self._init_occluded_factor = occluded_factor
        self._occluded_factor = occluded_factor

    def set_occluded_factor(self, occluded_factor: OccludedFactor):
        """ Override the current occluded factor instantiation of the environment."""
        self._occluded_factor = occluded_factor
        for element in occluded_factor.present_elements:
            self.agents[element.agent_id] = element
            self.initial_frame[element.agent_id] = element.state

    def reset(self):
        """ Reset the rollout to its initial state and removes occluded factors from the environment."""
        super().reset()
        self._occluded_factor = self._init_occluded_factor
