from typing import List, Dict, Union
from itertools import product

import logging
import igp2 as ip

from gofi.map.static_object import StaticObject

logger = logging.getLogger(__name__)


class OccludedFactor:
    """ Represents a set of binary occluded factors of arbitrary objects which are either present or not"""

    def __init__(self,
                 elements: List[ip.Agent],
                 presence: List[bool] = None,
                 force_visible: bool = False,
                 force_invisible: bool = False):
        """ Initialise a new occluded factor

        Args:
             elements: The elements (agents) that may be occluded in the environment.
             presence: The boolean presence of the factor corresponding to the elements at the same index.
                Must have the same length as positions.
            force_visible: Whether to force the agents to be visible (pz=1.) in the environment.
            force_invisible: Whether to force the agents to be invisible (pz=0.) in the environment.
        """

        self.__elements = elements
        if not elements:
            self.__presence = None
        else:
            assert presence is None or len(presence) == len(elements), \
                f"Number of objects is not equal to the number of presences"
            if not presence:
                self.__presence = [False] * len(elements)
            else:
                self.__presence = presence

        assert not (force_visible and force_invisible), "Cannot force both visible and invisible"
        self._force_visible = force_visible
        self._force_invisible = force_invisible

    def __repr__(self):
        object_reprs = ", ".join(map(repr, zip(self.__elements, self.__presence)))
        return f"OF({object_reprs})"

    def update_frame(self, frame: Dict[int, ip.AgentState], in_place: bool = False) -> Dict[int, ip.AgentState]:
        """ Update the state of the given frame by including the occluded elements that are in the factor.

        Args:
            frame: The frame to update.
            in_place: Whether to update the frame in place or return a new frame.
        """
        if in_place:
            for agent, present in zip(self.__elements, self.__presence):
                if present:
                    frame[agent.agent_id] = agent.state
            return frame
        else:
            new_frame = frame.copy()

            for agent, present in zip(self.__elements, self.__presence):
                if present:
                    new_frame[agent.agent_id] = agent.state
            return new_frame

    @classmethod
    def create_all_instantiations(
            cls,
            elements: List[ip.Agent],
            forced_visible_agents: List[int] = None
    ) -> List["OccludedFactor"]:
        """ Creates a list of factors with all combination of occluded element instantiations.
        Note, this function creates and array of size 2^len(elements)!

        Args:
            elements: Elements (agents) that may be occluded in the environment.
            forced_visible_agents: List of agent IDs that force the factor to be visible in the environment.
        """
        if elements:
            presences_product = product(*[[True, False]] * len(elements))
            if not forced_visible_agents:
                return [OccludedFactor(elements, presence) for presence in presences_product]
            else:
                ret = []
                forced_ids_idx = [i for i, a in enumerate(elements) if a.agent_id in forced_visible_agents]
                for presences in presences_product:
                    # Force visible if all forced agents are present
                    if all(presences[i] for i in forced_ids_idx):
                        ret.append(OccludedFactor(elements, presences, force_visible=True))
                    elif all(not presences[i] for i in forced_ids_idx):
                        ret.append(OccludedFactor(elements, presences, force_invisible=True))
                    else:
                        ret.append(OccludedFactor(elements, presences))
                return ret
        else:
            return [OccludedFactor([None], [False])]  # Empty factor if no elements present

    @property
    def elements(self) -> List[ip.Agent]:
        """ Returns the list of occluded elements in the environment. """
        return self.__elements

    @property
    def presence(self) -> List[bool]:
        """ Returns whether each occluded element in the factor is currently present or not. """
        return self.__presence

    @property
    def no_occlusions(self) -> bool:
        """ Whether there are no occlusions present in the environment. """
        return self.__presence is None or not any(self.__presence)

    @property
    def present_elements(self) -> List[ip.Agent]:
        """ Returns the list of occluded elements that are currently present in the environment. """
        return [agent for agent, present in zip(self.__elements, self.__presence) if present]

    @property
    def force_visible(self) -> bool:
        """ Whether the factor should be forced to be visible in the environment."""
        return self._force_visible

    @property
    def force_invisible(self) -> bool:
        """ Whether the factor should be forced to be invisible in the environment."""
        return self._force_invisible
