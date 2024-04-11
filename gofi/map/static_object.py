from typing import Optional

import igp2 as ip
import numpy as np
from shapely import Polygon


class StaticObject:
    """ Represent a static object on a Map with arbitrary shape defined by a boundary. """

    def __init__(self, boundary: Polygon, transparent: bool = None, object_type: str = None):
        """ Initialise a new static object.

         Args:
            boundary: The Polygon describing the shape boundary
            transparent: Whether the object is transparent or not
            object_type: The type of object the class represents (e.g., building, zebra crossing, etc.)
         """
        self.__boundary = boundary
        self.__transparent = transparent
        if transparent is None:
            self.__transparent = True if object_type in ["crossing"] else False
        self.__object_type = object_type

    @classmethod
    def from_description(cls, description: dict):
        """ Initialize a new object from a description. """
        transparent = description.get("transparent", None)
        otype = description["type"]
        if "shape" in description:
            if description["shape"] == "box":
                return cls(Polygon(ip.Box(**description["params"]).boundary), transparent, otype)
            else:
                raise AttributeError(f"Unsupported static object shape description: {description['shape']}")
        else:
            return cls(Polygon(description["vertices"]), transparent, otype)

    @property
    def boundary(self) -> Polygon:
        """ The boundary of the object. """
        return self.__boundary

    @property
    def boundary_coords(self) -> np.ndarray:
        """ Return the coordinate sequence of the boundary. Note ret[0]==ret[-1]. """
        return np.array(self.__boundary.exterior.coords)

    @property
    def center(self) -> np.ndarray:
        """ The center of mass of the polygon. """
        center = self.__boundary.centroid
        return np.array([center.x, center.y])

    @property
    def transparent(self) -> bool:
        """ Whether the object is see through i.e. transparent. """
        return self.__transparent

    @property
    def object_type(self) -> Optional[str]:
        """ The semantic type of the static object. """
        return self.__object_type
