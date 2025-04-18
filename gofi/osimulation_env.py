from typing import Dict, Any

import gymnasium as gym
import numpy as np
import igp2 as ip

from gofi.map.omap import OMap
from gofi.osimulation import OSimulation
from gofi.agents.gofi_agent import GOFIAgent
from gofi.agents.occluded_agent import OccludedAgent
from gofi.occluded_factor import StaticObject


MAX_ITERS = 10000

class OSimulationEnv(ip.simplesim.SimulationEnv):
    def __init__(self, config: Dict[str, Any], render_mode: str = None, max_iters: int = MAX_ITERS):
        """Initialise new simple simulation environment as a ParallelEnv.
        Args:
            config: Scenario configuration object.
            open_loop: If true then no physical controller will be applied.
        """
        self.config = config
        self.max_iters = max_iters

        # Set IPG2 configurations
        ip_config = ip.core.config.Configuration()
        ip_config.set_properties(**config["scenario"])
        ip.WaypointManeuver.ACC_ARGS["s_0"] = 3.0

        # Initialize simulation
        self.scenario_map = OMap.parse_from_description(config["scenario"]["map_path"], config.get("objects", []))
        self.fps = int(config["scenario"]["fps"]) if "fps" in config["scenario"] else 20
        self.open_loop = config["scenario"].get("open_loop", False)
        self.separate_ego = config["scenario"].get("separate_ego", False)
        self._simulation = OSimulation(self.scenario_map, self.fps, self.open_loop)

        # Set up Env variables
        self.n_agents = None
        self.reset_observation_space(init=True)
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
        )
        self.render_mode = render_mode

    def create_agent(self, agent_config, scenario_map, frame, fps):
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
        else:
            raise ValueError(f"Unsupported agent type {agent_config['type']}")
        return agent, rolename
