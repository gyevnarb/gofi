from typing import List, Tuple
import logging

import igp2 as ip

from gofi.occluded_factor import OccludedFactor

logger = logging.getLogger(__name__)


class OTree(ip.Tree):
    """ Tree structure to support distinguishing actions based on
    the presence of occluded factors. When calling select_plan()
    this tree uses a super-root to allow using existing code. """

    def __init__(self,
                 root: ip.Node,
                 action_policy: ip.Policy = None,
                 plan_policy: ip.Policy = None,
                 occluded_factors: List[OccludedFactor] = None):
        """ Create a super node to manage different driving behaviours based on occlusions. """
        super().__init__(root, action_policy, plan_policy)

        assert occluded_factors is not None, "Occluded factors must be provided. Else use a regular Tree."

        actions = ["Root" if of.no_occlusions else of for of in occluded_factors]
        super_root = ip.Node(("Super",), self._root.state.copy(), actions)
        super_root.expand()
        self._tree[("Super",)] = super_root
        self.add_child(super_root, self._root)
        self._root = super_root

    def set_occlusions(self, occluded_factor: OccludedFactor = None, allow_hide_occluded: bool = True) -> bool:
        """ Specifies which occlusions branch to use from the super node and performs necessary
         updates to bookkeeping.

        Args:
            occluded_factor: the occluded factor to use for the next simulation.
             allow_hide_occluded: whether to allow hiding the occluded factor in simulation.

        Returns:
             hide_occluded: whether the present occluded factor is hidden from the ego in simulation.
        """
        def pick_action():
            action_ = "Root" if occluded_factor.no_occlusions else occluded_factor
            idx_ = self.root.actions.index(action_)
            self.root.action_visits[idx_] += 1
            return action_

        self.root.state_visits += 1

        if occluded_factor.no_occlusions or self.root.action_visits[self.root.actions_names.index("Root")] == 0:
            # If "Root" has not yet been visited or there are no occlusion use deterministic action selection
            action = pick_action()
        else:
            # If there is an occlusion, use action policy to select branch. This will sometimes result in
            #  a rollout where the occluded factor is present but we intentionally hide it from the ego.
            action = self.select_action(self.root) if allow_hide_occluded else pick_action()
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

        hide_occluded = allow_hide_occluded and not occluded_factor.no_occlusions and action == "Root"
        return hide_occluded

    def backprop(self, r: float, final_key: Tuple, force_reward: bool = False):
        self._root = self._tree[("Super",)]
        super().backprop(r, final_key, force_reward)

    def select_plan(self) -> Tuple[List, Tuple[str]]:
        # Select the best super root node
        next_action, _ = self._plan_policy.select(self.root)
        if not isinstance(next_action, str): next_action = repr(next_action)

        # Find optimal action sequence from the current root then reset root to super-root
        self._root = self._tree[("Super",) + (next_action,)]
        plan, trace = super().select_plan()
        self._root = self._tree[("Super",)]

        return plan, trace

    def print(self, node: ip.Node = None):
        if node is None:
            node = self.tree[("Super", )]
        logger.debug(f"{node.key}: (A, Q)={list(zip(node.actions_names, node.q_values))}; Visits={node.action_visits}")
        for child in node.children.values():
            self.print(child)
