import logging

from shapely import LineString, Polygon

import igp2 as ip


logger = logging.getLogger(__name__)


class OSimulation(ip.simplesim.Simulation):
    def get_observations(self, agent_id: int = 0) -> ip.Observation:
        """ Get observations for the given agent. Occlusions are calculated based only on the line of sight as
        represented by a line without volume. Currently, this method applies occlusions only from the
        view of the ego agent (Agent ID 0).

        Args:
            agent_id: the id of the agent for which to generate the observation
        """
        if agent_id == 0:
            occluded_ids = [aid for aid, agent in self.agents.items()
                            if hasattr(agent, "is_occluded") and agent.is_occluded(self.t)]
            remove_occluded = {aid: state for aid, state in self.state.items() if aid not in occluded_ids}

            # Set the occluded state for each agent for the ego
            self.agents[agent_id].set_occluded_states(
                {aid: state for aid, state in self.state.items() if aid in occluded_ids})

            return ip.Observation(remove_occluded, self.scenario_map)
        else:
            return ip.Observation(self.state, self.scenario_map)
