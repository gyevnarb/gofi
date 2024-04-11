from typing import List, Dict, Union
from itertools import product

import logging
import igp2 as ip

from gofi.map.static_object import StaticObject

logger = logging.getLogger(__name__)


class OccludedFactor:
    """ Represents a set of binary occluded factors of arbitrary objects which are either present or not"""

    def __init__(self, elements: List[ip.Agent], presence: List[bool] = None):
        """ Initialise a new occluded factor

        Args:
             elements: The elements (agents) that may be occluded in the environment.
             presence: The boolean presence of the factor corresponding to the elements at the same index.
                Must have the same length as positions.
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

    def __repr__(self):
        object_reprs = ", ".join(map(repr, zip(self.__elements, self.__presence)))
        return f"OccludedFactor({object_reprs})"

    def update_frame(self, frame: Dict[int, ip.AgentState]) -> Dict[int, ip.AgentState]:
        """ Update the state of the given frame by including the occluded elements that are in the factor.

        Args:
            frame: The frame to update.
        """
        new_frame = frame.copy()

        for agent, present in zip(self.__elements, self.__presence):
            if present:
                new_frame[agent.agent_id] = agent.state
        return new_frame

    @classmethod
    def create_all_instantiations(cls, elements: list[ip.Agent]) -> list["OccludedFactor"]:
        """ Creates a list of factors with all combination of occluded element instantiations.
        Note, this function creates and array of size 2^len(elements)!

        Args:
            elements: Elements (agents) that may be occluded in the environment.
        """
        return [OccludedFactor(elements, presence) for presence in product(*[[True, False]] * len(elements))]

    @property
    def elements(self) -> list[ip.Agent]:
        """ Returns the list of occluded elements in the environment. """
        return self.__elements

    @property
    def presence(self) -> list[bool]:
        """ Returns whether each occluded element in the factor is currently present or not. """
        return self.__presence

    @property
    def no_occlusions(self) -> bool:
        """ Whether there are no occlusions present in the environment """
        return self.__elements is None

    @property
    def present_elements(self) -> list[ip.Agent]:
        """ Returns the list of occluded elements that are currently present in the environment. """
        return [agent for agent, present in zip(self.__elements, self.__presence) if present]
