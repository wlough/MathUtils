import numpy as np
import vedo
from pymathutils.mesh.pyutils import HalfEdgeMesh

m = HalfEdgeMesh.init_icososphere(
    num_refinements=1,
    compute_he_stuff=False,
)

xyz_coord_V = m.xyz_coord_V
V_cycle_F = m.V_cycle_F
vm = vedo.Mesh([xyz_coord_V, V_cycle_F])
vedo.show(vm, axes=1)
