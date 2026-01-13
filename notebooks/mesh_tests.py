from pymathutils.mesh.pyutils import (
    HalfEdgeMesh,
    HalfEdgePatch,
    SphericalHarmonicSurface,
    vf_samples_to_he_samples,
    tri_samples_to_combinatorialmap,
)

from pymathutils.mesh import (
    tri_vertex_cycles_to_half_edge_samples,
    find_halfedge_index_of_twin,
    load_vf_samples_from_ply,
)


from src.python.mesh_viewer import MeshViewer
import matplotlib.pyplot as plt
import numpy as np

m0 = HalfEdgeMesh.init_icososphere(
    num_refinements=0,
    compute_he_stuff=False,
)
he_samples = tri_vertex_cycles_to_half_edge_samples(m0.V_cycle_F)
he_samples = dict(he_samples) | {"xyz_coord_V": m0.xyz_coord_V}
m = HalfEdgeMesh.from_samples(**he_samples)
# m = HalfEdgeMesh.load_vf_ply("/home/wlough/projects/MeshBrane/data/example_ply/dumbbell_coarse_vf.ply")


# V_cycle_F = m0.V_cycle_F
# (
#     cmaps,
#     D_S0,
#     D_S1,
#     D_S2,
#     D,
#     Dplus,
#     Dminus,
#     S0,
#     S1,
#     S2,
#     enumeration_Dplus,
#     he_samples,
#     v_cycles,
# ) = tri_samples_to_combinatorialmap(V_cycle_F)
# %%

import pyvista as pv
import numpy as np
from pymathutils.mesh.pyutils import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer, get_half_edge_vector_field


# m = HalfEdgeMesh.init_icososphere(
#     num_refinements=2,
#     compute_he_stuff=True,
# )
m = HalfEdgeMesh.load_vf_ply("/home/wlough/projects/MeshBrane/data/example_ply/dumbbell_coarse_vf.ply")

xyz_coord_V = m.xyz_coord_V
V_cycle_F = m.V_cycle_F

pts_V, vecs_V = get_half_edge_vector_field(m)

pv_m = pv.PolyData(xyz_coord_V, faces=np.hstack([np.full((len(V_cycle_F), 1), 3), V_cycle_F]))

pv_pts = pv.PolyData(pts_V)
pv_pts["vecs"] = vecs_V  # Add vectors to mesh

# Plot mesh and vectors
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(pv_m, color="lightblue", show_edges=True)
plotter.add_arrows(pts_V, vecs_V, mag=1.0, color="red")  # Add arrows
plotter.show(jupyter_backend="static")


# %%

from pymathutils.mesh.pyutils import HalfEdgeMesh
from pymathutils.mesh import load_vf_samples_from_ply, write_vf_samples_to_ply

m = HalfEdgeMesh.init_icososphere(
    num_refinements=2,
    compute_he_stuff=True,
)

write_vf_samples_to_ply(m.xyz_coord_V, m.V_cycle_F, "output/icososphere2_vf.ply")

m = HalfEdgeMesh.load_vf_ply("/home/wlough/projects/MeshBrane/data/example_ply/dumbbell_coarse_vf.ply")

write_vf_samples_to_ply(m.xyz_coord_V, m.V_cycle_F, "output/dumbbell_coarse_vf.ply")
