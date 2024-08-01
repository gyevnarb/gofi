import numpy as np

import logging
from typing import Dict
import igp2 as ip

from gofi.recognition.ogoals_probabilities import OGoalsProbabilities

logger = logging.getLogger(__name__)


class OGoalRecognition(ip.GoalRecognition):
    """This class updates existing goal probabilities using likelihoods computed from the vehicle current trajectory.
    It also calculates the probabilities of up to n_trajectories paths to these goals. """

    def __init__(self,
                 astar: ip.AStar,
                 smoother: ip.VelocitySmoother,
                 scenario_map: ip.Map,
                 **kwargs):
        """Initialises a goal recognition class that will be used to update a OGoalProbabilities object.

        Args:
            astar: AStar object used to generate trajectories
            smoother: Velocity smoother object used to make the AStar generated trajectories realistic
            scenario_map: a Map object representing the current scenario

        Keyword Args:
            cost: a Cost object representing how the reward associated to each trajectory will be computed.
            beta: scaling parameter for the Boltzmann distribution generating the likelihoods
            gamma: scaling parameter for the Boltzmann distribution generating trajectory probabilities
            reward_as_difference: choose if we define the reward for each trajectory separately or if the
                                  reward is computed from differences of the different trajectory
                                  quantities alongside the pathlength.
            n_trajectories: The number of trajectories to try to return
        """
        super().__init__(astar, smoother, scenario_map,
                         kwargs.get("cost", None),
                         kwargs.get("n_trajectories", 1),
                         kwargs.get("beta", 1.),
                         kwargs.get("gamma", 1.),
                         kwargs.get("reward_as_difference", False))

    def update_goals_probabilities(self,
                                   goals_probabilities: OGoalsProbabilities,
                                   observed_trajectory: ip.Trajectory,
                                   agent_id: int,
                                   frame_ini: Dict[int, ip.AgentState],
                                   frame: Dict[int, ip.AgentState],
                                   visible_region: ip.Circle = None,
                                   debug: bool = False) \
            -> OGoalsProbabilities:
        """Updates the goal probabilities, and stores relevant information
        in the occluded GoalsProbabilities object.

        Args:
            goals_probabilities: occluded GoalsProbabilities object to update
            observed_trajectory: current vehicle trajectory
            agent_id: id of agent in current frame
            frame_ini: frame corresponding to the first state of the agent's trajectory
            frame: current frame
            visible_region: region of the map which is visible to the ego vehicle
            debug: Whether to plot A* planning
        """
        norm_factor = 0.
        current_lane = self._scenario_map.best_lane_at(frame[agent_id].position, frame[agent_id].heading)
        logger.info(f"Agent ID {agent_id} occluded goal recognition:")
        for factor in goals_probabilities.occluded_factors_probabilities:
            logger.debug(f"Factor {factor}:")
            factor_goal_sum = 0.
            oframe_ini = factor.update_frame(frame_ini)
            oframe = factor.update_frame(frame)
            for goal in goals_probabilities.goals_priors:
                key = (goal, factor)
                try:
                    logger.info(f"  Recognition for {factor} and {goal}")
                    if goal.reached(frame_ini[agent_id].position) and not isinstance(goal, ip.StoppingGoal):
                        raise RuntimeError(f"\tAgent {agent_id} reached goal at start.")

                    # Check if goal is not blocked by stopped vehicle
                    self._check_blocked(agent_id, current_lane, frame, goal)

                    # 4. and 5. Generate optimum trajectory from initial point and smooth it
                    if goals_probabilities.optimum_trajectory[key] is None:
                        logger.debug("\tGenerating optimum trajectory")
                        trajectories, plans = self._generate_trajectory(
                            1, agent_id, oframe_ini, goal,
                            state_trajectory=None, visible_region=visible_region, debug=debug)
                        goals_probabilities.optimum_trajectory[key] = trajectories[0]
                        goals_probabilities.optimum_plan[key] = plans[0]

                    opt_trajectory = goals_probabilities.optimum_trajectory[key]

                    # 7. and 8. Generate optimum trajectory from last observed point and smooth it
                    logger.debug(f"\tGenerating trajectory from current time step")
                    all_trajectories, all_plans = self._generate_trajectory(
                        self._n_trajectories, agent_id, oframe, goal, observed_trajectory,
                        visible_region=visible_region, debug=debug)

                    # 6. Calculate optimum reward
                    goals_probabilities.optimum_reward[key] = self._reward(opt_trajectory, goal)
                    logger.debug(f"\tOptimum costs: {self._cost.cost_components}")

                    # For each generated possible trajectory to this goal
                    for i, trajectory in enumerate(all_trajectories):
                        # join the observed and generated trajectories
                        trajectory.insert(observed_trajectory)

                        # 9,10. calculate rewards, likelihood
                        reward = self._reward(trajectory, goal)
                        logger.debug(f"\tT{i} costs: {self._cost.cost_components}")
                        goals_probabilities.all_rewards[key].append(reward)

                        reward_diff = self._reward_difference(opt_trajectory, trajectory, goal)
                        goals_probabilities.all_reward_differences[key].append(reward_diff)

                    # 11. Calculate likelihood
                    likelihood = self._likelihood(opt_trajectory, all_trajectories[0], goal)

                    # Calculate all trajectory probabilities
                    goals_probabilities.trajectories_probabilities[key] = \
                        self._trajectory_probabilities(goals_probabilities.all_rewards[key])

                    # Write additional goals probabilities fields
                    goals_probabilities.all_trajectories[key] = all_trajectories
                    goals_probabilities.all_plans[key] = all_plans
                    goals_probabilities.current_trajectory[key] = all_trajectories[0]
                    goals_probabilities.reward_difference[key] = \
                        goals_probabilities.all_reward_differences[key][0]
                    goals_probabilities.current_reward[key] = \
                        goals_probabilities.all_rewards[key][0]

                except RuntimeError as e:
                    logger.debug(str(e))
                    likelihood = 0.
                    goals_probabilities.current_trajectory[key] = None

                occluded_factors_prior = goals_probabilities.occluded_factors_priors[factor]
                if factor.force_visible:
                    occluded_factors_prior = 1.
                elif factor.force_invisible:
                    occluded_factors_prior = 0.

                # update goal probabilities
                goals_probabilities.likelihood[key] = likelihood
                goals_probabilities.goals_probabilities[key] = \
                    likelihood * \
                    goals_probabilities.goals_priors[goal] * \
                    occluded_factors_prior
                factor_goal_sum += goals_probabilities.goals_probabilities[key]

            goals_probabilities.occluded_factors_probabilities[factor] = factor_goal_sum
            norm_factor += factor_goal_sum

        # then divide prob by norm_factor to normalise
        for factor, pz_unnorm in goals_probabilities.occluded_factors_probabilities.items():
            try:
                pz = pz_unnorm
                goals_probabilities.occluded_factors_probabilities[factor] = pz / norm_factor
                for key, pg_z in goals_probabilities.goals_probabilities.items():
                    if key[1] == factor:
                        pgz = pg_z / pz if pz != 0. else goals_probabilities.goals_priors[key[0]]
                        goals_probabilities.goals_probabilities[key] = pgz
            except ZeroDivisionError:
                logger.debug("\tAll factors impossible. Setting probabilities to 0.")

        logger.debug(f"")
        logger.info("Final goals probabilities:")
        for factor, pz in goals_probabilities.occluded_factors_probabilities.items():
            logger.info(f"{factor}: {np.round(pz, 3)}")
            for key, pg_z in goals_probabilities.goals_probabilities.items():
                if pg_z != 0. and key[1] == factor:
                    logger.info(f"\t{key[0]}: {np.round(pg_z, 3)}")
        # logger.info()
        # logger.debug({k: v for k, v in goals_probabilities.goals_probabilities.items() if v != 0.})

        return goals_probabilities
