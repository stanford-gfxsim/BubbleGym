"""Shape feature extraction for bubble meshes.

Three modules:

- :mod:`shape_feature.mesh_utils` -- OBJ I/O, watertightness checks, connected
  components, orientation, signed volume / area / rescale, Mirtich mass
  properties, and Laplacian smoothing with an on-disk cache.
- :mod:`shape_feature.nonspherical_features` -- curvature integrals, convex-hull
  ratios, and the eight non-spherical features the frequency model consumes.
- :mod:`shape_feature.build_dataset` -- the parallel driver and CLI that
  computes every feature for a list of meshes in one pass per mesh.

Every feature here assumes a *closed* surface. ``load_obj_mesh`` validates that
on load and raises :class:`~shape_feature.mesh_utils.NonWatertightMeshError`
for a mesh with holes or non-manifold edges.
"""

from .mesh_utils import (
    MeshError,
    NonWatertightMeshError,
    check_watertight,
    compute_mirtich_moments,
    edge_incidence_report,
    ensure_outward_orientation,
    laplacian_smooth_vf,
    largest_connected_component_mesh,
    load_obj_mesh,
    load_or_smooth_obj,
    mesh_signed_volume,
    mesh_surface_area,
    principal_inertia,
    vertices_scaled_to_target_volume,
    write_obj,
)
from .nonspherical_features import (
    CHULL_FEATURE_COLS,
    FEATURE_COLS_8,
    INERTIA_FEATURE_COLS,
    NONSPH_CHULL_FEATURE_COLS,
    WADELL_FEATURE_COLS,
    InvalidSphericityError,
    compute_nonsphericity_columns,
    convex_hull_features_from_vf,
    convex_hull_vf,
    curvature_integrals_from_vf,
    mean_curvature_integral,
    nonspherical_features_from_unit_mesh,
    sphericities_from_integrals,
    wadell_phi,
    willmore_energy_vertex,
)

__all__ = [
    # mesh_utils
    "MeshError",
    "NonWatertightMeshError",
    "check_watertight",
    "compute_mirtich_moments",
    "edge_incidence_report",
    "ensure_outward_orientation",
    "laplacian_smooth_vf",
    "largest_connected_component_mesh",
    "load_obj_mesh",
    "load_or_smooth_obj",
    "mesh_signed_volume",
    "mesh_surface_area",
    "principal_inertia",
    "vertices_scaled_to_target_volume",
    "write_obj",
    # nonspherical_features
    "CHULL_FEATURE_COLS",
    "FEATURE_COLS_8",
    "INERTIA_FEATURE_COLS",
    "NONSPH_CHULL_FEATURE_COLS",
    "WADELL_FEATURE_COLS",
    "InvalidSphericityError",
    "compute_nonsphericity_columns",
    "convex_hull_features_from_vf",
    "convex_hull_vf",
    "curvature_integrals_from_vf",
    "mean_curvature_integral",
    "nonspherical_features_from_unit_mesh",
    "sphericities_from_integrals",
    "wadell_phi",
    "willmore_energy_vertex",
]
