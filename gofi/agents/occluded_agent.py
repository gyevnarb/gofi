from typing import Dict, List

import igp2 as ip


class OccludedAgent(ip.TrafficAgent):
    """ Agent executing a pre-defined macro action with a fixed set of times when it is occluded from the ego. """

    def __init__(self,
                 occlusions: List[Dict[str, float]],
                 agent_id: int,
                 initial_state: ip.AgentState,
                 goal: ip.Goal = None,
                 fps: int = 20,
                 macro_actions: List[ip.MacroAction] = None):
        """ Create a new occluded agent. """
        super().__init__(agent_id, initial_state, goal, fps, macro_actions)
        self._occlusions = occlusions

    def __repr__(self) -> str:
        return f"OccludedAgent(ID={self.agent_id})"

    def __str__(self) -> str:
        return self.__repr__()

    def is_occluded(self, t: int, observation: ip.Observation) -> bool:
        """ Check if the agent is occluded at the given time step. """
        current_occlusions = [occlusion for occlusion in self._occlusions
                             if occlusion["start"] <= t < occlusion["end"]
                             or "by_agent" in occlusion]
        if not current_occlusions:
            return False

        for occlusion in current_occlusions:
            if "by_agent" in occlusion and occlusion["by_agent"] not in observation.frame:
                continue

            if occlusion["start"] <= t < occlusion["end"]:
                return True

            if "by_agent" in occlusion:
                scenario_map = observation.scenario_map
                frame = observation.frame
                ego_lane = scenario_map.best_lane_at(frame[0].position, frame[0].heading)
                blocking_lane = scenario_map.best_lane_at(frame[occlusion["by_agent"]].position, frame[occlusion["by_agent"]].heading)
                occ_lane = scenario_map.best_lane_at(frame[self.agent_id].position, frame[self.agent_id].heading)
                if ego_lane == blocking_lane == occ_lane:
                    ego_pos = ego_lane.distance_at(frame[0].position)
                    blocking_pos = blocking_lane.distance_at(frame[occlusion["by_agent"]].position)
                    occ_pos = occ_lane.distance_at(frame[self.agent_id].position)
                    if ego_pos < blocking_pos < occ_pos:
                        return True
        return False

    @property
    def occlusions(self) -> List[Dict[str, float]]:
        """ The occlusions of the agent. """
        return self._occlusions
