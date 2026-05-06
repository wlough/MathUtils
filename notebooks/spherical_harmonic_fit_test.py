import sys
from pathlib import Path
import os

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

from pymathutils.mesh.pyutils import SphericalHarmonicSurface
import numpy as np

ply_path = f"{proj_dir}/data/example_ply/dumbbell.ply"

s0 = SphericalHarmonicSurface(
    # point_cloud_ply_path=elting_ply_paths[1],
    point_cloud_ply_path=ply_path,
    l_max=90,
    fit_type="xyz",
    reg_lambda=1e-6,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.05,
    # pre_surf_coord_transform=lambda xyz: xyz,
    pre_surf_coord_transform=lambda xyz: xyz,
    center_type="median",
    apply_cassini_flow=True,
    cassini_a=0.75,
    # cassini_center=np.array([0.0, 0.0, 0.0]),
    cassini_center=np.array([0.05, 0.0, -0.15]),
)
# s0.smooth_verts(num_iterations=80,
#                 xyz_condition=lambda xyz: np.linalg.norm(xyz) < 0.25)
# for _ in range(4):
#     s0.flip_non_delaunay()
# s0.plot_cloud_vs_mesh()

# %%

import matplotlib.pyplot as plt
from src_python.pybrane.spherical_harmonic_surface import (
    SphericalHarmonicSurface,
    geometric_median,
    newton_geometric_median,
    # apply_cassini_flow,
    # project_to_cassini_u,
    cassini_u,
    integrate_to_Gstar,
    step_with_tangent_projection,
    vectorized_integrate_to_Gstar,
)
import numpy as np

# ###########################
# center test
# %%
elting_ply_paths = [
    "./data/elting_data/mesh_t67_n0.ply",
    "./data/elting_data/mesh_t68_n0.ply",
    "./data/elting_data/mesh_t69_n0.ply",
    "./data/elting_data/mesh_t70_n0.ply",
    "./data/elting_data/mesh_t71_n0.ply",
    "./data/elting_data/mesh_t71_n1.ply",
]
# /home/wlough/projects/MeshBrane/output/dumbell_fits/dumbbell_he_spherical_harmonic_fit_lmax001.ply

s0 = SphericalHarmonicSurface(
    # point_cloud_ply_path=elting_ply_paths[1],
    point_cloud_ply_path="/home/wlough/projects/MeshBrane/output/division6/raw_data/envelope_004002.ply",
    l_max=90,
    fit_type="xyz",
    reg_lambda=1e-6,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.05,
    # pre_surf_coord_transform=lambda xyz: xyz,
    pre_surf_coord_transform=lambda xyz: xyz,
    center_type="median",
    apply_cassini_flow=True,
    cassini_a=0.75,
    # cassini_center=np.array([0.0, 0.0, 0.0]),
    cassini_center=np.array([0.05, 0.0, -0.15]),
)
# s0.smooth_verts(num_iterations=80,
#                 xyz_condition=lambda xyz: np.linalg.norm(xyz) < 0.25)
# for _ in range(4):
#     s0.flip_non_delaunay()
s0.plot_cloud_vs_mesh()
# %%
# s1 = SphericalHarmonicSurface(
#     point_cloud_ply_path=elting_ply_paths[1],
#     l_max=54,
#     fit_type="xyz",
#     reg_lambda=1e-3,
#     triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
#     icos_refinements=3,
#     triangulation_ply_path="",
#     # min_neighbor_dist=0.01,
#     pre_surf_coord_transform=lambda xyz: np.diag([2.0, 2.0, 1.0])
#     @ (xyz + [-0.05, 0.0, 0.15]),
# )
s0.xyz_coord_P.tolist()
tuple()
xyz = np.array(sorted(s0.xyz_coord_P.tolist(), key=lambda xyz: xyz[2]))

plt.plot(xyz[:0])
# s0.plot_cloud_vs_mesh()
# s1.plot_cloud_vs_mesh()
# a = 0.75
# pad = 2.2 / a
# u_lemniscate = np.log(a**2)
# u_convex = np.log(2 * a**2)
# eps = 1e-6
# u_list = np.linspace(u_lemniscate, u_convex, 4)
s0.cassini_a = 0.6
s0.plot_cloud_vs_cassini_ovaloid()
# s0.xyz_coord_P = apply_cassini_flow(s0.xyz_coord_P, a, dt=1e-4)
#
# X_old = s0.xyz_coord_P.copy()
# u_star = np.log(2.0 * a * a)
# X_new = project_to_cassini_u(X_old, a, u_star)
# X_new = vectorized_integrate_to_Gstar(X_old, a, u_star,
#                                       use_dG1=False,
#                                       rtol=1e-8, atol=1e-10,
#                                       max_step=np.inf)
# dU = np.array([abs(cassini_u(xyz, a) - u_star) for xyz in X_new])
# np.max(dU)
# s0.xyz_coord_P = X_new
#

# %%
xyz_coord_P = s0.xyz_coord_P
num_bins = 20

# def get_foci(xyz_coord_P, num_bins=10):
# X0 = [xyz for xyz in xyz_coord_P]
X0 = xyz_coord_P.copy()
# X = np.array(sorted(X0, key=lambda xyz: xyz[2]))
X = X0[np.argsort(X0[:, 2], kind="stable")]
x, y, z = X.T
zmin, zmax = z[0], z[-1] + 1e-12
dZ = zmax - zmin
zbin_bdrs = np.array([zmin + n * dZ / num_bins for n in range(num_bins + 1)])
zbin_rngs = np.column_stack([zbin_bdrs[:-1], zbin_bdrs[1:]])
bins = [[] for _ in range(num_bins)]
for i in range(len(X)):
    xyz = X[i]
    x, y, z = xyz
    bin_num = int((z - zmin) * num_bins / dZ)
    # print((z-zmin)*num_bins/dZ)
    z0, z1 = zbin_rngs[bin_num]
    is_in_bin = (z0 <= z) and (z < z1)
    if not is_in_bin:
        raise ValueError("AAAAA")
    # print(is_in_bin)
    # print(z)
    bins[bin_num].append([x, y, z])
ave_rho = []
ave_z = []
for bin in bins:
    x, y, z = np.array(bin).T
    ave_z_bin = np.mean(z)
    rho = np.sqrt(x**2 + y**2)
    ave_rho_bin = np.mean(rho)
    ave_rho.append(ave_rho_bin)
    ave_z.append(ave_z_bin)
    # print(ave_z_bin)

ave_rho = np.array(ave_rho)
ave_z = np.array(ave_z)
drho = ave_rho[1:] - ave_rho[:-1]
dz = ave_z[1:] - ave_z[:-1]
drho_dz = drho / dz
bin_order = np.argsort(drho_dz, kind="stable")
ave_z[bin_order]
# return np.array(ave_z), bin_order


# ave_z, bin_order = get_foci(
#     s0.xyz_coord_P, num_bins=10)
# ave_z[bin_order]
# ###########################
# smoothing and regularization test
# %%
elting_ply_paths = [
    "./data/elting_data/mesh_t67_n0.ply",
    "./data/elting_data/mesh_t68_n0.ply",
    "./data/elting_data/mesh_t69_n0.ply",
    "./data/elting_data/mesh_t70_n0.ply",
    "./data/elting_data/mesh_t71_n0.ply",
    "./data/elting_data/mesh_t71_n1.ply",
]
# no regularization
s0 = SphericalHarmonicSurface(
    point_cloud_ply_path=elting_ply_paths[1],
    l_max=11,
    fit_type="xyz",
    reg_lambda=0.0,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.01,
    pre_surf_coord_transform=lambda xyz: np.diag([2.0, 2.0, 1.0])
    @ (xyz + [-0.05, 0.0, 0.15]),
)

s1 = SphericalHarmonicSurface(
    point_cloud_ply_path=elting_ply_paths[1],
    l_max=54,
    fit_type="xyz",
    reg_lambda=1e-3,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.01,
    pre_surf_coord_transform=lambda xyz: np.diag([2.0, 2.0, 1.0])
    @ (xyz + [-0.05, 0.0, 0.15]),
)

s2 = SphericalHarmonicSurface(
    point_cloud_ply_path=elting_ply_paths[1],
    l_max=54,
    fit_type="xyz",
    reg_lambda=1e-3,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.01,
    pre_surf_coord_transform=lambda xyz: np.diag([2.0, 2.0, 1.0])
    @ (xyz + [-0.05, 0.0, 0.15]),
)


s2.smooth_verts(num_iterations=80, xyz_condition=lambda xyz: np.linalg.norm(xyz) < 0.25)
for _ in range(4):
    s2.flip_non_delaunay()


s0.plot_cloud_vs_mesh()
s1.plot_cloud_vs_mesh()
s2.plot_cloud_vs_mesh()


# ###########################
# Cartesian fit Elting for potato mesh
# %%
elting_ply_paths = [
    "./data/elting_data/mesh_t67_n0.ply",
    "./data/elting_data/mesh_t68_n0.ply",
    "./data/elting_data/mesh_t69_n0.ply",
    "./data/elting_data/mesh_t70_n0.ply",
    "./data/elting_data/mesh_t71_n0.ply",
    "./data/elting_data/mesh_t71_n1.ply",
]
s = SphericalHarmonicSurface(
    point_cloud_ply_path=elting_ply_paths[4],
    l_max=50,
    fit_type="xyz",
    reg_lambda=1e-5,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.0,
    pre_surf_coord_transform=lambda xyz: np.diag([1.0, 1.0, 1.0])
    @ (xyz + np.array([-0.05, 0.0, 0.15])),
)

s.plot_cloud_vs_mesh()
# ###########################
# Cartesian fit dumbbell
# %%

s = SphericalHarmonicSurface(
    point_cloud_ply_path="./data/example_ply/dumbbell_fine_vf.ply",
    l_max=30,
    fit_type="xyz",
    reg_lambda=1e-5,
    triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
    icos_refinements=3,
    triangulation_ply_path="",
    min_neighbor_dist=0.0,
    pre_surf_coord_transform=lambda xyz: np.diag([1.0, 1.0, 1.0])
    @ (xyz + np.array([0.0, 0.0, 0.0])),
)

s.plot_cloud_vs_mesh()
