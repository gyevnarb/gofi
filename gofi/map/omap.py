import igp2 as ip
import logging

from gofi.map.static_object import StaticObject

logger = logging.getLogger(__name__)


class OMap(ip.Map):
    def __init__(self, opendrive: ip.opendrive.OpenDrive = None, objects: list[StaticObject] = None):
        """ Create a map object given a parsed OpenDrive file and static object descriptions

        Args:
            opendrive: A class describing the parsed contents of the OpenDrive file.
            objects: A list of static objects in the environment.
        """
        super().__init__(opendrive)
        self.__objects = objects

    @classmethod
    def parse_from_description(cls, file_path: str, static_objects: list[dict]):
        """ Parse the OpenDrive file and create a new Map instance

        Args:
            file_path: The absolute/relative path to the OpenDrive file
            static_objects: A list of objects in the environment

        Returns:
            A new instance of the Map class
        """
        new_map = cls.parse_from_opendrive(file_path)
        new_map.__objects = [StaticObject.from_description(object_description) for object_description in static_objects]
        return new_map

    @property
    def objects(self) -> list[StaticObject]:
        """ Retrieve all objects in the map. """
        return self.__objects

    @property
    def buildings(self) -> list[StaticObject]:
        """ Return all buildings in the environment. """
        return [obj for obj in self.__objects if obj.object_type == "building"]
