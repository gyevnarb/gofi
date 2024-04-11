import igp2 as ip

from gofi.agents.occluded_agent import OccludedAgent


class Pedestrian(OccludedAgent):
    """ Agent representing a pedestrian."""

    def __repr__(self):
        return f"Pedestrian(ID={self.agent_id})"
