import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
from shapely.ops import unary_union

import emopt

def _closed_to_open(coords):
    """Remove repeated last == first if present."""
    if len(coords) >= 2 and coords[0] == coords[-1]:
        return coords[:-1]
    return coords

def _shift_xy(xs, ys, x_off: float, y_off: float):
    """Apply (x_off, y_off) to coordinate lists."""
    return [float(x) + x_off for x in xs], [float(y) + y_off for y in ys]

def _poly_to_xy_lists_shifted(poly: ShapelyPolygon, x_off: float, y_off: float):
    """Return (xs, ys) for emopt from a Shapely polygon exterior with frame shift applied."""
    coords = _closed_to_open(list(poly.exterior.coords))
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return _shift_xy(xs, ys, x_off, y_off)

def _emit_polygon_with_holes_shifted(
    eps_struct: emopt.grid.StructuredMaterial2D,
    poly: ShapelyPolygon,
    layer_outer: int,
    eps_outer: complex | float,
    layer_holes: int,
    eps_hole: complex | float,
    x_off: float,
    y_off: float,
):
    """Add a polygon (outer ring) and carve its holes (inner rings) with a frame shift."""
    # Outer
    xs, ys = _poly_to_xy_lists_shifted(poly, x_off, y_off)
    outer = emopt.grid.Polygon(xs=xs, ys=ys)
    outer.layer = layer_outer
    outer.material_value = eps_outer
    eps_struct.add_primitive(outer)

    # Holes (overwrite to background or other ε)
    for interior in poly.interiors:
        ring = _closed_to_open(list(interior.coords))
        hxs = [p[0] for p in ring]
        hys = [p[1] for p in ring]
        hxs, hys = _shift_xy(hxs, hys, x_off, y_off)
        hole = emopt.grid.Polygon(xs=hxs, ys=hys)
        hole.layer = layer_holes
        hole.material_value = eps_hole
        eps_struct.add_primitive(hole)

def _emit_geometry_shifted(
    eps_struct: emopt.grid.StructuredMaterial2D,
    geom,
    layer_solid: int,
    eps_solid: complex | float,
    layer_hole: int,
    eps_hole: complex | float,
    x_off: float,
    y_off: float,
):
    """
    Handle Polygon, MultiPolygon, or iterable of polygons; shift into [0,W]×[0,H].
    """
    if geom is None:
        return

    if isinstance(geom, ShapelyPolygon):
        _emit_polygon_with_holes_shifted(
            eps_struct, geom, layer_solid, eps_solid, layer_hole, eps_hole, x_off, y_off
        )
        return

    if isinstance(geom, ShapelyMultiPolygon):
        for g in geom.geoms:
            _emit_polygon_with_holes_shifted(
                eps_struct, g, layer_solid, eps_solid, layer_hole, eps_hole, x_off, y_off
            )
        return

    # Try generic iterable (e.g., list of polygons)
    try:
        for g in geom:
            _emit_geometry_shifted(
                eps_struct, g, layer_solid, eps_solid, layer_hole, eps_hole, x_off, y_off
            )
    except TypeError:
        # Not iterable and not a polygon -> ignore
        pass


# ------------------------------
# Public bridge
# ------------------------------

def build_eps_from_generator(
    vcsel_generator,
    dx: float,
    dy: float,
    pad: float,
    eps_mesa: complex | float,
    eps_aperture: complex | float,
    eps_background: complex | float = 1,
):
    """
    Create (eps, mu, domain) for emopt from a VCSELGenerator, with frame-safe shifting.

    Coordinates:
      - Domain uses the *world frame* [x0..x1] × [y0..y1].
      - Geometry primitives are shifted by (-x0, -y0) so they lie in [0,W] × [0,H].

    Parameters
    ----------
    dx, dy : grid steps (same units as your geometry, e.g., μm)
    pad    : scalar padding around the device bounding box (same units)
    eps_background : ε for free space (≈1.0)
    eps_mesa       : ε for oxidized GaAs (low index)
    eps_aperture   : ε for GaAs-like core (high index)
    include_aperture : if True, paint aperture after mesa so it overwrites

    Returns
    -------
    eps : emopt.grid.StructuredMaterial2D
    mu  : emopt.grid.ConstantMaterial2D
    domain : emopt.misc.DomainCoordinates
    """

    # 1) World-space bbox from generator (minx,miny,maxx,maxy)
    minx, miny, maxx, maxy = vcsel_generator.bounding_box
    x0 = float(minx) - pad
    x1 = float(maxx) + pad
    y0 = float(miny) - pad
    y1 = float(maxy) + pad

    W = x1 - x0
    H = y1 - y0

    # 3) Structured epsilon & constant mu
    eps = emopt.grid.StructuredMaterial2D(H, W, dx, dy)
    mu = emopt.grid.ConstantMaterial2D(1.0)

    # 4) Background rectangle (already in shifted frame)
    bg = emopt.grid.Rectangle(x0=0.0, y0=0.0, xspan=W, yspan=H)
    bg.layer = 4
    bg.material_value = eps_background
    eps.add_primitive(bg)

    # 5) Mesa (overwrite background), holes carved back to background
    _emit_geometry_shifted(
        eps_struct=eps,
        geom=getattr(vcsel_generator, "_mesa", None),
        layer_solid=3,
        eps_solid=eps_mesa,
        layer_hole=2,
        eps_hole=eps_background,
        x_off=0,
        y_off=0,
    )

    # 6) Optional aperture (painted last so it overwrites mesa)
    _emit_geometry_shifted(
        eps_struct=eps,
        geom=vcsel_generator._aperture,
        layer_solid=1,
        eps_solid=eps_aperture,
        layer_hole=0,
        eps_hole=eps_background,
        x_off=0,
        y_off=0,
    )

    # 7) Domain in world coords (unchanged API)
    domain = emopt.misc.DomainCoordinates(
        x0, x1,  # world frame
        y0, y1,
        0.0, 0.0,  # z extents unused in 2D
        dx, dy, 1.0
    )

    return eps, mu, domain