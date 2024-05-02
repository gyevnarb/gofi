import logging
import sys

import igp2 as ip
import gofi
import numpy as np
import random
import matplotlib.pyplot as plt

from util import generate_random_frame, parse_args, load_config, to_ma_list, setup_logging
from igp2.core.config import Configuration

logger = logging.getLogger(__name__)


def main():
    args, config, scenario_map, frame = init()

    if args.plot_map_only:
        gofi.plot_map(scenario_map, hide_road_bounds_in_junction=True, markings=True)
        for aid, state in frame.items():
            plt.plot(*state.position, marker="o")
        plt.show()
        return True

    simulation = None
    try:
        simulation, result = run_simple_simulation(frame, scenario_map, args, config)
    except Exception as e:
        logger.exception(msg=str(e), exc_info=e)
        result = False
    finally:
        if simulation is not None:
            del simulation

    return result


def run_simple_simulation(frame, scenario_map, args, config):
    fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20
    simulation = gofi.OSimulation(scenario_map, fps)

    for agent_config in config["agents"]:
        agent, rolename = create_agent(agent_config, scenario_map, frame, fps, args)
        simulation.add_agent(agent, rolename=rolename)

    if args.plot:
        ip.simplesim.plot_simulation(simulation, debug=False, map_plotter=gofi.plot_map)
        plt.show()

    for t in range(config["scenario"]["max_steps"]):
        simulation.step()
        if args.plot is not None and t % args.plot == 0:
            ip.simplesim.plot_simulation(simulation, debug=False, map_plotter=gofi.plot_map)
            plt.show()
    return simulation, True


def create_agent(agent_config, scenario_map, frame, fps, args):
    base_agent = {"agent_id": agent_config["id"], "initial_state": frame[agent_config["id"]],
                  "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])), "fps": fps}

    mcts_agent = {"scenario_map": scenario_map,
                  "cost_factors": agent_config.get("cost_factors", None),
                  "view_radius": agent_config.get("view_radius", None),
                  "kinematic": not args.carla,
                  "velocity_smoother": agent_config.get("velocity_smoother", None),
                  "goal_recognition": agent_config.get("goal_recognition", None),
                  "stop_goals": agent_config.get("stop_goals", False)}

    agent_type = agent_config["type"]

    if agent_type == "GOFIAgent":
        agent = gofi.GOFIAgent(**base_agent, **mcts_agent, **agent_config["mcts"])
        rolename = "ego"
    elif agent_type in "TrafficAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        rolename = agent_config.get("rolename", "car")
        agent = ip.TrafficAgent(**base_agent)
    elif agent_type == "OccludedAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        agent = gofi.OccludedAgent(occlusions=agent_config["occlusions"], **base_agent)
        rolename = agent_config.get("rolename", "occluded")
    elif agent_type == "Pedestrian":
        agent = gofi.Pedestrian(occlusions=agent_config["occlusions"], **base_agent)
        rolename = agent_config.get("rolename", "pedestrian")
    elif agent_type == "StaticObject":
        agent = gofi.StaticObject(**base_agent)
        rolename = agent_config.get("rolename", "object")
    else:
        raise ValueError(f"Unsupported agent type {agent_config['type']}")
    return agent, rolename


def init():
    args = parse_args()
    config = load_config(args)

    setup_logging(debug=args.debug, log_path=args.save_log_path)

    logger.debug(args)

    seed = args.seed if args.seed else config["scenario"]["seed"] if "seed" in config["scenario"] else 21

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip_config = Configuration()
    ip_config.set_properties(**config["scenario"])

    xodr_path = config["scenario"]["map_path"]
    scenario_map = gofi.OMap.parse_from_description(xodr_path, config.get("objects", []))

    frame = generate_random_frame(scenario_map, config)
    return args, config, scenario_map, frame


if __name__ == '__main__':
    sys.exit(main())
