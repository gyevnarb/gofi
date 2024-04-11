"""
Copied directly from the OGRIT source code:
Link: https://github.com/uoe-agents/OGRIT/blob/master/ogrit/occlusion_detection/visualisation_tools.py
A collection of methods used to plot the maps.
"""

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import igp2 as ip
import numpy as np

from gofi.map.polygon_patch import PolygonPatch
from gofi.map.omap import OMap


# Set up the colors for the plots.
OCCLUSIONS_COLOR = "g"
OCCLUDED_LANE_COLOR = "r"
OBSTACLES_COLOR = "y"

OCCLUDED_AREA_ALPHA = 0.7
OCCLUSION_SEGMENTS_ALPHA = 0.3


def get_box(vehicle, x=None, y=None, heading=None):
    """
    Get the boundaries of the vehicle.

    If x, y and heading are given, use the vehicle only for its metadata (length and width).
    """

    if x is not None and y is not None and heading is not None:
        return ip.Box(np.array([x, y]),
                      vehicle.metadata.length,
                      vehicle.metadata.width,
                      heading)

    return ip.Box(np.array([vehicle.position[0],
                            vehicle.position[1]]),
                  vehicle.metadata.length,
                  vehicle.metadata.width,
                  vehicle.heading)


def plot_map(scenario_map: OMap, ax: plt.Axes = None, scenario_config: dict = None, frame=None, **kwargs):
    """ Draws the road layout of a Map object with static objects and plots over that the occluded regions from
    the viewpoint of the ego vehicle (Agent ID 0).

    Keyword Args:
        object_fill_color: What color to fill static objects (default: black)
    """
    if kwargs is None:
        kwargs = {"midline": False, "markings": False}

    if scenario_config is not None:
        ax = ip.plot_map(scenario_map, ax, scenario_config=scenario_config,
                         ignore_roads=True, plot_goals=False, plot_buildings=True, **kwargs)
    else:
        ax = ip.plot_map(scenario_map, ax, **kwargs)

        for obj in scenario_map.objects:
            coords = np.array(obj.boundary.exterior.coords.xy)
            ax.plot(*coords, color=kwargs.get("object_fill_color", "k"))
            if obj.object_type == "building":
                ax.fill_between([coords[0].min(), coords[0].max()], coords[1].min(), coords[1].max(),
                                color=kwargs.get("object_fill_color", "k"))

            elif obj.object_type == "crossing":
                # TODO: Only works properly for horizontal roads!
                x = [coords[0].min(), coords[0].max()]
                y = coords[1].min()
                n = 5
                dy = (coords[1].max() - y) / (2 * n)
                for i in range(2 * n):
                    c = kwargs.get("object_fill_color", "k" if i % 2 == 0 else "white")
                    ax.fill_between(x, y, y + dy, color=c)
                    y += dy

    if frame:
        for aid, state in frame.items():
            ax.text(*state.position, aid)
            plot_area(Polygon(get_box(state).boundary), color=OBSTACLES_COLOR, ax=ax)
    return ax


def plot_area(polygon, color="r", alpha=.5, linewidth=None, ax=None):
    """
    Given a polygon, plot the boundaries and shade the interior.
    """
    if ax is None:
        ax = plt.gca()
    patch = PolygonPatch(polygon, color=color, alpha=alpha, fill=True, linewidth=linewidth)
    ax.add_patch(patch)


def plot_occlusions(ego_position: np.array, occlusion_lines: List[List[Tuple[int, int]]] = None,
                    road_occlusions: Dict[int, Dict[int, List[Polygon]]] = None, non_visible_areas: Polygon = None):
    """
    Plot the occlusions on the map.

    Args:
        ego_position:      position of the vehicle from whose perspective the occlusions are taken
        occlusion_lines:   lines from the ego vehicle to each of the obstacles. It should include 4 points
                           (represented as tuples) v1, v2, v3, v4 as described in the OGRIT paper Section 4.3
        road_occlusions:   dictionary with the road id as index and another dictionary as value. This latter
                           dictionary has the road's lane id as index and the occlusions (as polygons) as values.
        non_visible_areas: polygon containing the areas that are never visible to the ego even if there are no
                           occlusions (e.g., because they are too far away).
    """

    plot_ego_position(ego_position)

    if occlusion_lines is not None:
        # Plot the line that go from the ego to the obstacles and shade the area occluded by the obstacle.
        occluded_areas = []
        for occlusion_line in occlusion_lines:
            x0, y0 = ego_position

            # v1=(x1, y1) and v2=(x2, y2) are the vertices of the obstacle. The other two vertices are the extensions.
            # Please, refer to Section 4.3 of the OGRIT paper for more information.
            ((x1, y1), (x3, y3)), ((x2, y2), (x4, y4)) = occlusion_line

            # Plot the line segment going from the ego vehicle to the vertices of the obstacle.
            plt.plot([x0, x1], [y0, y1], color=OCCLUSIONS_COLOR, alpha=OCCLUDED_AREA_ALPHA)
            plt.plot([x0, x2], [y0, y2], color=OCCLUSIONS_COLOR, alpha=OCCLUDED_AREA_ALPHA)

            occluded_areas.append(Polygon([(x1, y1), (x2, y2), (x4, y4), (x3, y3)]))

        if non_visible_areas is not None:
            # Plot the areas that are far away and are no visible to the ego even if there are no obstacles.
            occluded_areas.append(non_visible_areas)

        plot_area_from_list(occluded_areas, color=OCCLUSIONS_COLOR, alpha=OCCLUDED_AREA_ALPHA)

    if road_occlusions is None:
        return

    # Plot the areas on the lanes that are occluded.
    lane_occlusions_all = []
    for road_id, occlusions in road_occlusions.items():
        if road_id == "occlusions":
            continue

        for lane_id, lane_occlusions in occlusions.items():

            if lane_id == "occlusions" or lane_occlusions is None:
                continue

            lane_occlusions_all.append(lane_occlusions)
    plot_area_from_list(lane_occlusions_all, color=OCCLUDED_LANE_COLOR, alpha=OCCLUDED_AREA_ALPHA)


def plot_area_from_list(geometries, color="r", alpha=0.5, ax=None):
    if ax is None:
        ax = plt.gca()
    geometries = unary_union(geometries)
    if not geometries.is_empty:
        if isinstance(geometries, Polygon):
            plot_area(geometries, color=color, alpha=alpha, ax=ax)
        elif isinstance(geometries, MultiPolygon):
            for geometry in geometries.geoms:
                plot_area(geometry, color=color, alpha=alpha, ax=ax)


def plot_ego_position(ego_position):
    plt.plot(*ego_position, marker="x")


def show_plot():
    plt.show()