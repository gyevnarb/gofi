import igp2 as ip


class OccludedAgent(ip.MacroAgent):
    """ Agent executing a pre-defined macro action with a fixed set of times when it is occluded from the ego. """

    def __init__(self,
                 occlusions: list[dict[str, float]],
                 agent_id: int,
                 initial_state: ip.AgentState,
                 goal: ip.Goal = None,
                 fps: int = 20):
        """ Create a new occluded agent. """
        super().__init__(agent_id, initial_state, goal, fps)
        self._occlusions = occlusions

    def __repr__(self):
        return f"OccludedAgent(ID={self.agent_id})"

    def is_occluded(self, t: int) -> bool:
        """ Check if the agent is occluded at the given time step. """
        return any(occlusion["start"] <= t <= occlusion["end"] for occlusion in self._occlusions)

    @property
    def occlusions(self) -> list[dict[str, float]]:
        """ The occlusions of the agent. """
        return self._occlusions
