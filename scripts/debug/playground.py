import sys

import gofi
import igp2 as ip


def main() -> int:
    scenario_map = ip.Map.parse_from_opendrive("scenarios/configs/scenario4.xodr")



if __name__ == '__main__':
    sys.exit(main())
