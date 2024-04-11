import sys

import gofi
import igp2 as ip


def main() -> int:
    goals = [ip.PointGoal([1, 1]), ip.PointGoal([2, 2])]
    factors = [gofi.OccludedFactor([[5, 5]]), gofi.OccludedFactor([[3, 3]])]
    gp = gofi.OGoalsProbabilities(goals, factors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
