from typing import Dict, Any
import igp2 as ip

from gofi.osimulation import OSimulation
from gofi.agents.gofi_agent import GOFIAgent
from gofi.agents.occluded_agent import OccludedAgent
from gofi.occluded_factor import StaticObject



class OSimulationEnv(ip.simplesim.SimulationEnv):
    def __init__(self, config: Dict[str, Any], render_mode: str = None):
        """Initialise the simulation environment with occluded factors."""
        super().__init__(config, render_mode)
        self._simulation = OSimulation(self.scenario_map, self.fps, self.open_loop)

    def create_agent(self, agent_config, scenario_map, frame, fps, args):
        base_agent = {
            "agent_id": agent_config["id"],
            "initial_state": frame[agent_config["id"]],
            "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])),
            "fps": fps,
        }

        mcts_agent = {
            "scenario_map": scenario_map,
            "cost_factors": agent_config.get("cost_factors", None),
            "view_radius": agent_config.get("view_radius", None),
            "kinematic": True,
            "velocity_smoother": agent_config.get("velocity_smoother", None),
            "goal_recognition": agent_config.get("goal_recognition", None),
            "stop_goals": agent_config.get("stop_goals", False),
            "occluded_factors_prior": agent_config.get("occluded_factors_prior", 0.1),
        }

        if agent_config["type"] == "GOFIAgent":
            agent = GOFIAgent(**base_agent, **mcts_agent, **agent_config["mcts"])
            rolename = "ego"
        elif agent_config["type"] in "TrafficAgent":
            if "macro_actions" in agent_config and agent_config["macro_actions"]:
                base_agent["macro_actions"] = self._to_ma_list(
                    agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
            rolename = agent_config.get("rolename", "car")
            agent = ip.TrafficAgent(**base_agent)
        elif agent_config["type"] == "OccludedAgent":
            if "macro_actions" in agent_config and agent_config["macro_actions"]:
                base_agent["macro_actions"] = self._to_ma_list(
                    agent_config["macro_actions"],
                    agent_config["id"],
                    frame,
                    scenario_map,
                )
            agent = OccludedAgent(
                occlusions=agent_config["occlusions"], **base_agent
            )
            rolename = agent_config.get("rolename", "occluded")
        elif agent_config["type"] == "StaticObject":
            agent = StaticObject(**base_agent)
            rolename = agent_config.get("rolename", "object")
        else:
            raise ValueError(f"Unsupported agent type {agent_config['type']}")
        return agent, rolename
