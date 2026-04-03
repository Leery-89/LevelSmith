"""
Building footprint and layout utilities.

Provides functions for creating and manipulating building footprints.
Migrated from generate_level.py for better modularity.
"""

import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.affinity import translate as shapely_translate


def make_rect_footprint(w=12.0, d=8.0):
    """
    Rectangular footprint (basic rectangle, for backward compatibility).
    
    Args:
        w: Width (X-axis)
        d: Depth (Z-axis)
    
    Returns:
        Shapely Polygon
    """
    return shapely_box(0, 0, w, d)


def make_l_footprint(w=12.0, d=8.0, cut_frac_x=0.45, cut_frac_z=0.45):
    """
    L-shaped footprint: cut a rectangle from the top-right corner.
    
    Args:
        w: Width (X-axis)
        d: Depth (Z-axis)
        cut_frac_x: Fraction of width to cut
        cut_frac_z: Fraction of depth to cut
    
    Returns:
        Shapely Polygon
    """
    rect = shapely_box(0, 0, w, d)
    cut  = shapely_box(w * (1 - cut_frac_x), d * (1 - cut_frac_z), w, d)
    return rect.difference(cut)


def make_u_footprint(w=12.0, d=8.0, notch_frac_x=0.4, notch_frac_z=0.55):
    """
    U-shaped footprint: cut a rectangle from the center bottom (opening forward).
    
    Args:
        w: Width (X-axis)
        d: Depth (Z-axis)
        notch_frac_x: Fraction of width for the notch (centered)
        notch_frac_z: Fraction of depth for the notch (from front edge)
    
    Returns:
        Shapely Polygon
    """
    rect = shapely_box(0, 0, w, d)
    nx0  = w * (0.5 - notch_frac_x / 2)
    nx1  = w * (0.5 + notch_frac_x / 2)
    cut  = shapely_box(nx0, 0, nx1, d * notch_frac_z)
    return rect.difference(cut)


def rotate_footprint(footprint, angle_deg, center=(0, 0)):
    """
    Rotate a footprint around a center point.
    
    Args:
        footprint: Shapely Polygon
        angle_deg: Rotation angle in degrees
        center: Rotation center (x, y)
    
    Returns:
        Rotated Shapely Polygon
    """
    from shapely.affinity import rotate
    return rotate(footprint, angle_deg, origin=center, use_radians=False)


def translate_footprint(footprint, dx, dy):
    """
    Translate (move) a footprint.
    
    Args:
        footprint: Shapely Polygon
        dx: X translation
        dy: Y translation
    
    Returns:
        Translated Shapely Polygon
    """
    return shapely_translate(footprint, xoff=dx, yoff=dy)


def scale_footprint(footprint, scale_x, scale_y, origin=(0, 0)):
    """
    Scale a footprint.
    
    Args:
        footprint: Shapely Polygon
        scale_x: X scale factor
        scale_y: Y scale factor
        origin: Scaling origin
    
    Returns:
        Scaled Shapely Polygon
    """
    from shapely.affinity import scale
    return scale(footprint, xfact=scale_x, yfact=scale_y, origin=origin)


def footprint_area(footprint):
    """
    Calculate area of a footprint.
    
    Args:
        footprint: Shapely Polygon
    
    Returns:
        Area in square units
    """
    return footprint.area


def footprint_bounds(footprint):
    """
    Get bounding box of a footprint.
    
    Args:
        footprint: Shapely Polygon
    
    Returns:
        Tuple (minx, miny, maxx, maxy)
    """
    return footprint.bounds


def footprints_overlap(footprint1, footprint2):
    """
    Check if two footprints overlap.
    
    Args:
        footprint1: First Shapely Polygon
        footprint2: Second Shapely Polygon
    
    Returns:
        True if footprints intersect
    """
    return footprint1.intersects(footprint2)


def compute_footprint_centroid(footprint):
    """
    Compute centroid of a footprint.
    
    Args:
        footprint: Shapely Polygon
    
    Returns:
        Tuple (x, y) of centroid
    """
    centroid = footprint.centroid
    return (centroid.x, centroid.y)


def create_footprint_grid(rows, cols, footprint_func=make_rect_footprint, 
                          spacing_x=16.0, spacing_z=12.0, **footprint_kwargs):
    """
    Create a grid of footprints.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        footprint_func: Function to create individual footprint
        spacing_x: X spacing between footprints
        spacing_z: Z spacing between footprints
        **footprint_kwargs: Arguments passed to footprint_func
    
    Returns:
        List of (footprint, position) tuples
    """
    footprints = []
    
    for row in range(rows):
        for col in range(cols):
            x = col * spacing_x
            z = row * spacing_z
            footprint = footprint_func(**footprint_kwargs)
            footprint = translate_footprint(footprint, x, z)
            footprints.append((footprint, (x, z)))
    
    return footprints