import logging

import igp2 as ip
from gofi.agents.occluded_agent import OccludedAgent
from gofi.agents.gofi_agent import GOFIAgent


logger = logging.getLogger(__name__)


class OSimulation(ip.simplesim.Simulation):
    def get_observations(self, agent_id: int = 0) -> ip.Observation:
        """ Get observations for the given agent. Occlusions are calculated based only on the line of sight as
        represented by a line without volume. Currently, this method applies occlusions only from the
        view of the ego agent (Agent ID 0).

        Args:
            agent_id: the id of the agent for which to generate the observation
        """
        full_observation = super().get_observations(agent_id)
        if agent_id == 0 and isinstance(self.agents[agent_id], GOFIAgent):
            occluded_ids = [aid for aid, agent in self.agents.items() if agent is not None
                            and isinstance(agent, OccludedAgent) and agent.is_occluded(self.t, full_observation)]

            remove_occluded = {aid: state for aid, state in self.state.items() if state is not None
                               and aid not in occluded_ids}

            force_visible = [aid for aid, agent in self.agents.items() if agent is not None
                             and isinstance(agent, OccludedAgent)
                             and not agent.is_occluded(self.t, full_observation)]

            # Set the occluded state for each agent for the ego at the start of the simulation
            if self.t == 0:
                self.agents[agent_id].set_occluded_states(
                    {aid: state for aid, state in self.state.items() if aid in occluded_ids})
            self.agents[agent_id].force_visible_agents(force_visible)

            return ip.Observation(remove_occluded, self.scenario_map)
        else:
            return full_observation
