from typing import Dict, Any
import igp2 as ip

from gofi.osimulation import OSimulation

class OSimulationEnv(ip.simplesim.SimulationEnv):
    def __init__(self, config: Dict[str, Any], render_mode: str = None):
        """ Initialise the simulation environment with occluded factors."""
        super().__init__(config, render_mode)
        self._simulation = OSimulation(self.scenario_map, self.fps, self.open_loop)
