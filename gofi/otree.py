from typing import List, Dict, Tuple
import copy

import igp2 as ip
from gofi.occluded_factor import OccludedFactor
from gofi.ogoals_probabilities import OGoalsProbabilities


class OTree(ip.Tree):
    """ Tree structure to support distinguishing actions based on
    the presence of occluded factors. When calling select_plan()
    this tree uses a super-root to allow using existing code. """

    def __init__(self,
                 root: ip.Node,
                 action_policy: ip.Policy = None,
                 plan_policy: ip.Policy = None,
                 predictions: Dict[int, OGoalsProbabilities] = None):
        """ Create a super node to manage different driving behaviours based on occlusions. """
        super().__init__(root, action_policy, plan_policy, predictions)
        actions = ["Root" if of.no_occlusions else of
                   for of in list(predictions.values())[0].occluded_factors]
        super_root = ip.Node(("Super",), self._root.state.copy(), actions)
        super_root.expand()
        self._tree[("Super",)] = super_root
        self.add_child(super_root, self._root)
        self._root = super_root

    def set_occlusions(self, occluded_factor: OccludedFactor = None):
        """ Specifies which occlusions branch to use from the super node and performs necessary
         updates to bookkeeping.

         Returns:
             hide_occluded: whether the present occluded factor is hidden from the ego in simulation.
             """
        self.root.state_visits += 1

        if occluded_factor.no_occlusions or self.root.action_visits[self.root.actions_names.index("Root")] == 0:
            # If "Root" has not yet been visited or there are no occlusion use deterministic action selection
            action = "Root" if occluded_factor.no_occlusions else occluded_factor
            idx = self.root.actions.index(action)
            self.root.action_visits[idx] += 1
        else:
            # If there is an occlusion, use action policy to select branch. This will sometimes result in
            #  a rollout where the occluded factor is present but we intentionally hide it from the ego.
            action = self.select_action(self.root)
        key = ("Super", str(action))

        if key in self._tree:
            self._root = self._tree[key]
        else:
            root = self._tree[("Super", "Root")]
            new_state = occluded_factor.update_frame(root.state)
            new_node = ip.Node(key, new_state, root.actions.copy())
            new_node.expand()
            self.add_child(self.root, new_node)
            self._root = new_node

        hide_occluded = not occluded_factor.no_occlusions and action == "Root"
        return hide_occluded

    def backprop(self, r: float, final_key: Tuple):
        self._root = self._tree[("Super",)]
        super().backprop(r, final_key)

    def select_plan(self) -> List:
        next_action, _ = self._plan_policy.select(self.root)
        if not isinstance(next_action, str): next_action = repr(next_action)
        self._root = self._tree[("Super",) + (next_action,)]
        return super().select_plan()
