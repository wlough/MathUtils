from .pyhalf_edge_mesh import HalfEdgeMesh
from ..mathutils_backend import thetaphi_from_xyz
from ..mathutils_backend.special import (
    compute_all_real_Ylm,
    # spherical_harmonic_index_n_LM,
    spherical_harmonic_index_lm_N,
    fit_real_sh_coefficients_to_points,
)
from ..jit_pyutils import fib_sphere
import numpy as np
from numba import njit

# ###################################################
# ###################################################
# ###################################################
# ###################################################


@njit
def point_cloud_principal_components(xyz):
    """
    # ***

    Principal components of the point cloud.

    ex=pc3, ey=pc2, ez=pc1.
    """

    # Calculate the covariance matrix
    cov_matrix = np.cov(xyz, rowvar=False)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Get the indices of the largest eigenvalues
    principal_indices = np.argsort(eigenvalues)
    # Return the corresponding eigenvectors
    ex, ey, ez = eigenvectors[:, principal_indices].T
    # Ensure ez/ex has positive z/x-component
    if ez[2] < 0:
        ez = -ez
    if ex[0] < 0:
        ex = -ex
    ezx = np.cross(ez, ex)
    ezx_dot_ey = ezx[0] * ey[0] + ezx[1] * ey[1] + ezx[2] * ey[2]
    # Ensure the right-handed coordinate system
    if ezx_dot_ey < 0:
        ey = -ey
    return ex, ey, ez


@njit
def thin_point_cloud_random(xyz_points, min_dist, max_iterations=1000):
    """
    # ***
    Thin point cloud by removing points that are less than min_dist apart.
    """
    n_points = len(xyz_points)
    if n_points == 0:
        return np.zeros(0, dtype=np.int64)

    keep_mask = np.ones(n_points, dtype=np.bool_)
    min_dist_sqr = min_dist * min_dist

    # Random order for processing
    indices = np.arange(n_points)
    np.random.shuffle(indices)
    removed_any = False
    for iteration in range(max_iterations):
        # print(f"iteration {iteration}               ", end="\r")
        print(f"iteration {iteration}               ")
        removed_any = False

        for idx_i in range(n_points):
            i = indices[idx_i]
            if not keep_mask[i]:
                continue

            # Check against all other kept points
            for idx_j in range(idx_i + 1, n_points):
                j = indices[idx_j]
                if not keep_mask[j]:
                    continue

                dx = xyz_points[i, 0] - xyz_points[j, 0]
                dy = xyz_points[i, 1] - xyz_points[j, 1]
                dz = xyz_points[i, 2] - xyz_points[j, 2]
                dist_sqr = dx * dx + dy * dy + dz * dz

                if dist_sqr < min_dist_sqr:
                    keep_mask[j] = False
                    removed_any = True

        if not removed_any:
            break

    return np.where(keep_mask)[0]


def geometric_median(
    P, w=None, x0=None, tol=1e-9, max_iter=500, eps=1e-12, return_info=False
):
    """
    Weighted geometric median via (modified) Weiszfeld.
    P : (N,d) points
    w : (N,) nonnegative weights (default: ones)
    x0: initial guess (default: weighted mean)
    tol: relative step tolerance
    eps: distance clamp to avoid division-by-zero
    """
    P = np.asarray(P, dtype=float)
    N, d = P.shape
    if w is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    if not np.all(w >= 0):
        raise ValueError("weights must be nonnegative")
    if np.sum(w) == 0:
        raise ValueError("sum of weights is zero")

    if x0 is None:
        x = (w[:, None] * P).sum(axis=0) / np.sum(w)  # weighted mean
    else:
        x = np.asarray(x0, dtype=float).copy()

    for it in range(max_iter):
        R = P - x[None, :]
        dists = np.linalg.norm(R, axis=1)

        # If we hit (or are extremely close to) a data point, check optimality
        j = np.argmin(dists)
        if dists[j] < eps:
            # Test if that data point is optimal (subgradient condition)
            vec = P[j] - P
            dj = np.linalg.norm(vec, axis=1)
            mask = np.arange(N) != j
            g = (w[mask, None] * (vec[mask] / np.maximum(dj[mask], eps)[:, None])).sum(
                axis=0
            )
            if np.linalg.norm(g) <= w[j]:
                info = {"iters": it, "stopped_at_point": True}
                return (P[j].copy(), info) if return_info else P[j].copy()
            # Not optimal: take a tiny step away and continue
            step_dir = -g
            nrm = np.linalg.norm(step_dir)
            if nrm == 0:
                # fallback: jitter in a random direction
                step_dir = np.random.randn(d)
                step_dir /= np.linalg.norm(step_dir)
            x = P[j] + (10.0 * eps) * step_dir / np.linalg.norm(step_dir)
            continue

        # Weiszfeld update with distance clamp
        inv = w / np.maximum(dists, eps)
        x_new = (inv[:, None] * P).sum(axis=0) / inv.sum()

        # Convergence test (relative)
        step = np.linalg.norm(x_new - x)
        if step <= tol * max(1.0, np.linalg.norm(x)):
            info = {"iters": it + 1, "stopped_at_point": False}
            return (x_new, info) if return_info else x_new
        x = x_new

    info = {"iters": max_iter, "stopped_at_point": False, "warning": "max_iter reached"}
    return (x, info) if return_info else x


#
#
# def newton_geometric_median(
#     P, W=None, x0=None, tol=1e-9, max_iter=500, eps=1e-12, return_info=False
# ):
#     """uses Newton's method to compute weighted geometric median"""
#     P = np.asarray(P, dtype=float)
#     N, d = P.shape
#     if W is None:
#         W = np.ones(N, dtype=float)
#     else:
#         W = np.asarray(W, dtype=float)
#     if not np.all(W >= 0):
#         raise ValueError("weights must be nonnegative")
#     if np.sum(W) == 0:
#         raise ValueError("sum of weights is zero")
#
#     if x0 is None:
#         x = (W[:, None] * P).sum(axis=0) / np.sum(W)  # weighted mean
#     else:
#         x = np.asarray(x0, dtype=float).copy()
#
#     I3 = np.eye(3)
#
#     def gradE_hessE(xyz):
#         dE = np.zeros(3)
#         ddE = np.zeros((3, 3))
#
#         for w, p in zip(W, P):
#             xp = x - p
#             norm_xp = np.linalg.norm(xp)
#             if norm_xp < 1e-12:
#                 norm_xp = 1e-12
#             dE += w * (x - p) / norm_xp
#             ddE += w * (I3 - np.outer(xp, xp) / (norm_xp * norm_xp)) / norm_xp
#         return dE, ddE
#
#     for iter in range(max_iter):
#         dE, dEE = gradE_hessE(x)
#         dx = np.linalg.solve(dEE, -dE)
#         x_new = x + dx
#         norm_dx = np.linalg.norm(dx)
#         if norm_dx <= tol * max(1.0, np.linalg.norm(x_new)):
#             info = {"iters": iter + 1, "stopped_at_point": False}
#             return (x_new, info) if return_info else x_new
#         x = x_new
#
#     info = {"iters": max_iter, "stopped_at_point": False, "warning": "max_iter reached"}
#     return (x, info) if return_info else x
#
#
# def uvphi_cassini_from_xyz(xyz, a):
#     """
#     Inverse map: (x,y,z) -> (u,v,phi) for Cassini-ovaloid coords with foci at (0,0,±a).
#     xyz : (N,3) array
#     a   : >0
#     """
#     xyz = np.asarray(xyz, dtype=float)
#     x, y, z = xyz.T
#     rho = np.hypot(x, y)  # row-wise sqrt(x^2+y^2)
#     w_re = z * z - rho * rho - a * a
#     w_im = 2.0 * z * rho
#     u = 0.5 * np.log(w_re * w_re + w_im * w_im)  # = log|w|
#     v = np.arctan2(w_im, w_re)  # = arg(w)
#     phi = np.arctan2(y, x)
#     return np.column_stack([u, v, phi])
#
#
# def xyz_from_cassini_uvphi(uvphi, a):
#     """
#     Forward map: (u,v,phi) -> (x,y,z).
#     uvphi : (N,3) array with u∈R, v∈(-π,π], phi∈(-π,π]
#     a     : >0
#     """
#     uvphi = np.asarray(uvphi, dtype=float)
#     u, v, phi = uvphi.T
#     eu = np.exp(u)
#     alpha = a * a + eu * np.cos(v)
#     beta = eu * np.sin(v)
#     # R = |a^2 + e^{u+iv}| = sqrt(alpha^2 + beta^2), computed stably:
#     R = np.hypot(alpha, beta)
#     # Meridional coords (ρ ≥ 0 by construction; z sign follows sign(beta))
#     rho_sq = 0.5 * (R - alpha)
#     z_sq = 0.5 * (R + alpha)
#     rho = np.sqrt(np.maximum(rho_sq, 0.0))
#     z = np.sign(beta) * np.sqrt(np.maximum(z_sq, 0.0))
#     # (When beta == 0, this picks z >= 0; if you need continuity across v=0,π, choose the sign consistently per track.)
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return np.column_stack([x, y, z])
#
#
# # def choose_cassini_foci(xyz_coord_P):
# # choose
#
#
# def cassini_ovaloid_level_set_fun(x, y, z, a, u):
#     """
#     zero for points xyz with the given cassini_uvphi u value
#     """
#     # x, y, z = xyz
#     rho2 = x**2 + y**2
#     iso_val = (z**2 - rho2 - a**2) ** 2 + 4 * z**2 * rho2 - np.exp(2 * u)
#     return iso_val
#
#
# def cassini_ovaloid_level_set_time(x, y, z, a):
#     """
#     zero for points xyz with the given cassini_uvphi u value
#     """
#     # x, y, z = xyz
#     rho2 = x**2 + y**2
#     exp2u = (z**2 - rho2 - a**2) ** 2 + 4 * z**2 * rho2
#     t = np.sqrt(exp2u)
#     return t
#
#
# def scalar_grad_cassini_ovaloid_level_set_fun(x, y, z, a):
#     rho = np.hypot(x, y)  # row-wise sqrt(x^2+y^2)
#     phi = np.arctan2(y, x)
#     rho2 = rho * rho
#     z2 = z * z
#     vel_rho = 8 * rho * z2 - 4 * (z2 - rho2 - a**2) * rho
#     xdot = vel_rho * np.cos(phi)
#     ydot = vel_rho * np.sin(phi)
#     zdot = 8 * rho2 * z - 4 * (z2 - rho2 - a**2) * z
#     return xdot, ydot, zdot
#
#
# def grad_cassini_ovaloid_level_set_fun(xyz_coord_P, a):
#     xyzdot_coord_P = np.zeros_like(xyz_coord_P)
#     x, y, z = xyz_coord_P.T
#     rho = np.hypot(x, y)  # row-wise sqrt(x^2+y^2)
#     phi = np.arctan2(y, x)
#     rho2 = rho * rho
#     z2 = z * z
#     vel_rho = 8 * rho * z2 - 4 * (z2 - rho2 - a**2) * rho
#     xyzdot_coord_P[:, 0] = vel_rho * np.cos(phi)
#     xyzdot_coord_P[:, 1] = vel_rho * np.sin(phi)
#     xyzdot_coord_P[:, 2] = 8 * rho2 * z - 4 * (z2 - rho2 - a**2) * z
#     return xyzdot_coord_P
#
#
# def apply_cassini_flow(xyz_coord_P0, a, dt=1e-4):
#     xyz_coord_P = xyz_coord_P0.copy()
#     u_lemniscate = np.log(a**2)
#     t_lemniscate = a**2
#     u_convex = np.log(2 * a**2)
#     t_convex = 2 * a**2
#     t_stop = t_convex
#     uvphi_coord_P = uvphi_cassini_from_xyz(xyz_coord_P, a)
#     u0 = uvphi_coord_P[:, 0]
#     T0 = np.exp(u0)
#     X = xyz_coord_P.copy()
#     num_points = len(X)
#     for n in range(num_points):
#         print(f"\n n={n}")
#         x, y, z = X[n]
#         t = T0[n]
#         print(f"t_start = {t}")
#
#         while t < t_stop:
#             # print(f"t={t}", end="\r")
#
#             xdot, ydot, zdot = scalar_grad_cassini_ovaloid_level_set_fun(x, y, z, a)
#             x += dt * xdot
#             y += dt * ydot
#             z += dt * zdot
#             tt = cassini_ovaloid_level_set_time(x, y, z, a)
#             print(f"tt-t={tt-t}", end="\r")
#             t += dt
#         X[n] = np.array([x, y, z])
#     return X
#
#
# # # ----- Cassini scalar and "u" -----
# # def cassini_G(xyz, a):
# #     """
# #     G(x,y,z) = (z^2 - (x^2+y^2) - a^2)^2 + 4 z^2 (x^2+y^2) = e^{2u}
# #     """
# #     xyz = np.asarray(xyz, float)
# #     x, y, z = xyz.T
# #     r2 = x*x + y*y
# #     w_re = z*z - r2 - a*a
# #     G = w_re*w_re + 4.0*z*z*r2
# #     return G
# #
# #
# # def cassini_u(xyz, a):
# #     """u = 0.5 * log G"""
# #     G = cassini_G(xyz, a)
# #     return 0.5 * np.log(np.maximum(G, 1e-300))
# #
# #
# # # ----- Exact Cartesian gradient of G -----
# # def grad_cassini_G(xyz, a):
# #     """
# #     ∇G = (4x(S+a^2), 4y(S+a^2), 4z(S-a^2)), with S = x^2+y^2+z^2
# #     """
# #     xyz = np.asarray(xyz, float)
# #     x, y, z = xyz.T
# #     S = x*x + y*y + z*z
# #     gx = 4.0 * x * (S + a*a)
# #     gy = 4.0 * y * (S + a*a)
# #     gz = 4.0 * z * (S - a*a)
# #     return np.column_stack([gx, gy, gz])
# #
# #
# # # ----- Project points to the Cassini level u_target (monotone, stable) -----
# # def project_to_cassini_u(xyz0, a, u_target,
# #                          eta=0.8, alpha_max=0.25, tol=1e-12, max_iter=200):
# #     """
# #     Move each point along ∇G to reach G* = exp(2 u_target).
# #     Uses per-point step length α = η (G* - G)/||∇G||^2 (clipped).
# #     Monotone in G and typically converges in ~5–20 steps.
# #     """
# #     xyz = np.asarray(xyz0, float).copy()
# #     Gstar = np.exp(2.0 * u_target)
# #     eps = 1e-30
# #
# #     for it in range(max_iter):
# #         G = cassini_G(xyz, a)
# #         gap = Gstar - G
# #         done = gap <= tol * np.maximum(Gstar, 1.0)
# #         if np.all(done):
# #             break
# #
# #         g = grad_cassini_G(xyz, a)
# #         g2 = np.einsum('ij,ij->i', g, g) + eps
# #
# #         # gap-controlled step (only where not done)
# #         alpha = eta * gap / g2
# #         alpha = np.clip(alpha, 0.0, alpha_max)
# #         xyz += (alpha[:, None] * g)
# #
# #     return xyz
#
#
# # G(x) = e^{2u} for the Cassini-ovaloid family (foci at (0,0,±a))
#
#
# def cassini_G(x, a):
#     x, y, z = x[..., 0], x[..., 1], x[..., 2]
#     r2 = x * x + y * y
#     w_re = z * z - r2 - a * a
#     return w_re * w_re + 4.0 * z * z * r2
#
#
# def cassini_u(xyz, a):
#     """u = 0.5 * log G"""
#     G = cassini_G(xyz, a)
#     return 0.5 * np.log(np.maximum(G, 1e-300))
#
#
# def grad_cassini_G(x, a):
#     x, y, z = x[..., 0], x[..., 1], x[..., 2]
#     S = x * x + y * y + z * z
#     gx = 4.0 * x * (S + a * a)
#     gy = 4.0 * y * (S + a * a)
#     gz = 4.0 * z * (S - a * a)
#     return np.stack([gx, gy, gz], axis=-1)
#
#
# def field_unit(x, a, eps=1e-30):
#     g = grad_cassini_G(x, a)
#     gnorm = np.maximum(np.linalg.norm(g), eps)
#     return g / gnorm
#
#
# def field_unit_dG1(x, a, eps=1e-30):
#     g = grad_cassini_G(x, a)
#     g2 = max(np.dot(g, g), eps)
#     return g / g2  # dG/dτ = 1
#
#
# def step_with_tangent_projection(x, a, h, n_corr=1):
#     """
#     One step along ∇G with length h, then project the chord to be parallel
#     to ∇G at the endpoint (enforces endpoint tangency).
#     """
#     gk = grad_cassini_G(x, a)
#     x_pred = x + h * gk
#     x_new = x_pred
#     for _ in range(n_corr):
#         g1 = grad_cassini_G(x_new, a)
#         dx = x_new - x
#         # project dx onto g1
#         coef = np.dot(dx, g1) / max(np.dot(g1, g1), 1e-30)
#         x_new = x + coef * g1
#     return x_new
#
#
# def get_foci(xyz_coord_P, num_bins=10):
#     X0 = [xyz for xyz in xyz_coord_P]
#     X = np.array(sorted(X0, key=lambda xyz: xyz[2]))
#     x, y, z = X.T
#     zmin, zmax = z[0], z[-1]
#     dZ = zmax - zmin
#     zbin_bdrs = np.array([zmin + n * dZ / num_bins for n in range(num_bins + 1)])
#
#     bins = num_bins * [[]]
#     for xyz in X:
#         x, y, z = xyz
#         bin_num = int((z - zmin) * num_bins / dZ)
#         bins[bin_num].append(xyz)


class SphericalHarmonicSurface(HalfEdgeMesh):
    """
    V -- vertex indices
    F -- face indices
    E -- edge indices
    H -- half-edge indices
    B -- boundary indices
    P -- processedpoint cloud indices
    N -- spherical harmonic mode indices. (l, m)->l(l+1)+m

    Input
    -----
    raw_point_cloud -- Cartesian coordinates of the unprocessed point cloud.
    l_max -- maximum spherical harmonic degree to use
    reg_lambda -- regularization parameter for the spherical harmonics

    *** process raw data to get nice point cloud
    xyz_coord_P -- Cartesian coords of points in the cloud
    r_coord_P -- radius coordinates of points in the cloud

    *** assign surface coordinates to point cloud and evaluate spherical harmonics
    surf_coord_P -- surface coordinates of points in the point cloud
    sph_harm_PN -- spherical harmonic basis functions evaluated at the point cloud surface coordinates

    *** Solve for spherical harmonic coefficients
    coeff_xyz_N -- expansion coefficients for xyz coordinates
    coeff_r_N -- expansion coefficients for radius coordinates

    *** assign triangulation of the unit sphere and evaluate spherical harmonics
    V_cycle_F -- face cycle indices for the triangulation
    surf_coord_V -- surface coordinates of vertices in the mesh
    sph_harm_VN -- spherical harmonic basis functions evaluated at the mesh vertices

    *** compute Cartesian coordinates of vertices from spherical harmonic coefficients
    xyz_coord_V -- xyz coordinates of vertices in the mesh = sph_harm_VN @ coeff_xyz_N




    Properties
    ----------
    raw_point_cloud : np.ndarray
        The raw point cloud data.
    origin : np.ndarray
        The origin of the point cloud, computed as the mean of the raw point cloud.
    ex, ey, ez : np.ndarray
        Orthonormal basis defined by the principal components (3rd,2nd,1st) of the point cloud.

    xyz_point_cloud : np.ndarray
        xyz coordinates of the point cloud expressed in the frame defined by origin, ex, ey, ez. Scaled to fit inside a unit sphere.
    rthetaphi_point_cloud : np.ndarray
        Spherical coordinates (r, theta=polar angle, phi=azimuthal angle) of the xyz_point_cloud.
    l_max : int
        The maximum value to use for l index of spherical harmonics Ylm
    ply_save_path : str
        The path to save the PLY file.
    coefficients_save_path : str
        The path to save the spherical harmonic coefficients.
    coefficients : np.ndarray of complex128
        The spherical harmonic coefficients for the surface.

    HalfEdgeMesh attributes: xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H, h_right_F, h_negative_B, V_cycle_F, V_cycle_E,...

    Methods
    -------
    transform_to_principal_frame()
        Set xyz_point_cloud to be the point cloud expressed in the principal component frame with coordinates scaled to fit inside a unit sphere.
    set_coefficients_to_unit_sphere()
        Set coefficients to represent a unit sphere.
    update_rthetaphi_from_xyz()
        Update the rthetaphi_point_cloud from the xyz_point_cloud.

    set_mesh_to_icososphere(num_refinements=0)
        Set HalfEdgeMesh data to an icosahedron with the specified number of refinements.

    """

    def __init__(
        self,
        raw_point_cloud=np.zeros((0, 3), dtype=np.float64),
        point_cloud_ply_path="",
        l_max=30,
        fit_type="xyz",
        reg_lambda=1e-5,
        triangulation_type="icos",  # "icos", "vf_ply", "he_ply"
        icos_refinements=2,
        triangulation_ply_path="",
        min_neighbor_dist=0.0,
        center_type="mean",
        pre_surf_coord_transform=lambda xyz: xyz,
        apply_cassini_flow=False,
        cassini_a=0.75,
        cassini_center=np.array([0.0, 0.0, 0.0]),
    ):
        super().__init__(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
        self.point_cloud_ply_path = point_cloud_ply_path
        if point_cloud_ply_path != "":
            print(f"Reading point cloud from {point_cloud_ply_path}...")
            try:
                m = HalfEdgeMesh.from_vf_ply(
                    point_cloud_ply_path, compute_he_stuff=False
                )
            except Exception as e:
                print(f"Error reading vf ply file {point_cloud_ply_path}: {e}")
                print("trying he ply...")
                m = HalfEdgeMesh.from_he_ply(
                    point_cloud_ply_path, compute_vf_stuff=False
                )
            self.raw_point_cloud = m.xyz_coord_V.copy()
        else:
            self.raw_point_cloud = raw_point_cloud.copy()
        self.l_max = l_max
        self.fit_type = fit_type
        self.reg_lambda = reg_lambda
        self.triangulation_type = triangulation_type
        self.icos_refinements = icos_refinements
        self.triangulation_ply_path = triangulation_ply_path
        self.min_neighbor_dist = min_neighbor_dist
        self.pre_surf_coord_transform = pre_surf_coord_transform
        self.center_type = center_type
        self.apply_cassini_flow = apply_cassini_flow
        self.cassini_a = cassini_a
        self.cassini_center = cassini_center
        if len(self.raw_point_cloud) == 0:
            return

        # ***
        self.xyz_coord_P = self.get_standardized_xyz_coord_P()
        if self.min_neighbor_dist > 0:
            print("Thinning point cloud...")
            keep_indices = thin_point_cloud_random(
                self.xyz_coord_P, self.min_neighbor_dist, max_iterations=1000
            )
            self.xyz_coord_P = self.xyz_coord_P[keep_indices].copy()
        self.set_triangulation()  # and surf_coord_V
        self.fit_to_xyz_coord_P()

    # ###################################################
    # ###################################################
    # ###################################################
    # ###################################################
    # ###################################################

    def get_standardized_xyz_coord_P(self):
        """
        Apply
        Applies rigid transformation + scaling so that:
          - the centroid of the point cloud is at (0,0,0)
          - the (1st, 2nd, 3rd) principal components are aligned with the
            coordinate axes (ez, ey, ex)
          - the point cloud is scaled to fit inside a unit sphere.
        """
        print("Transforming point cloud to principal frame...")
        # Center the point cloud
        if self.center_type == "mean":
            center = np.mean(self.raw_point_cloud, axis=0)
        elif self.center_type == "median":
            # center = np.mean(self.raw_point_cloud, axis=0)
            center = geometric_median(
                self.raw_point_cloud,
                w=None,
                x0=center,
                tol=1e-9,
                max_iter=500,
                eps=1e-12,
                return_info=False,
            )
        else:
            raise ValueError("center type must be mean or median")
        xyz_coord_P = self.raw_point_cloud - center
        ex, ey, ez = point_cloud_principal_components(xyz_coord_P)
        xyz_coord_P = np.einsum("sj, ij->si", xyz_coord_P, [ex, ey, ez])
        r_coord_P = np.linalg.norm(xyz_coord_P, axis=1)
        rmax = np.max(r_coord_P)
        xyz_coord_P /= rmax
        # r_coord_P /= rmax
        return xyz_coord_P

    def set_triangulation(self):
        if self.triangulation_type == "icos":
            print(
                f"Setting triangulation to icosahedron with {self.icos_refinements} refinements..."
            )
            m = HalfEdgeMesh.init_icososphere(num_refinements=self.icos_refinements)
        elif self.triangulation_type == "vf_ply":
            print(f"Reading triangulation from {self.triangulation_ply_path}...")
            m = HalfEdgeMesh.from_vf_ply(
                self.triangulation_ply_path, compute_he_stuff=True
            )
        elif self.triangulation_type == "he_ply":
            print(f"Reading triangulation from {self.triangulation_ply_path}...")
            m = HalfEdgeMesh.from_he_ply(
                self.triangulation_ply_path, compute_vf_stuff=True
            )

        self.h_out_V = m.h_out_V
        self.v_origin_H = m.v_origin_H
        self.h_next_H = m.h_next_H
        self.h_twin_H = m.h_twin_H
        self.f_left_H = m.f_left_H
        self.h_right_F = m.h_right_F
        self.h_negative_B = m.h_negative_B
        self.V_cycle_F = m.V_cycle_F
        self.V_cycle_E = m.V_cycle_E

        self.surf_coord_V = thetaphi_from_xyz(m.xyz_coord_V)

    def fit_to_xyz_coord_P(self):
        self.r_coord_P = np.linalg.norm(self.xyz_coord_P, axis=1)
        # ***
        self.surf_coord_P = self.get_surf_coord_P()
        self.sph_harm_PN = compute_all_real_Ylm(self.l_max, self.surf_coord_P)
        # ***
        self.coeff_xyz_N = self.solve_for_coeff_xyz_N()
        # self.coeff_xyz_N = fit_real_sh_coefficients_to_points(
        #     self.xyz_coord_P, self.l_max, self.reg_lambda)
        # ***
        self.sph_harm_VN = compute_all_real_Ylm(self.l_max, self.surf_coord_V)
        self.xyz_coord_V = self.sph_harm_VN @ self.coeff_xyz_N

    def solve_for_coeff_xyz_N(self):
        print("Solving for spherical harmonic coefficients...")
        num_modes = self.sph_harm_PN.shape[1]
        little_l = np.array(
            [spherical_harmonic_index_lm_N(n)[0] for n in range(num_modes)]
        )
        L = np.diag(little_l * (little_l + 1))  # Regularization term
        A = self.sph_harm_PN.T @ self.sph_harm_PN + self.reg_lambda * L.T @ L
        b = self.sph_harm_PN.T @ self.xyz_coord_P
        A_inv = np.linalg.inv(A)
        coeff_xyz_N = A_inv @ b
        return coeff_xyz_N
        # return fit_real_sh_coefficients_to_points(
        #     self.xyz_coord_P, self.l_max, self.reg_lambda
        # )

    ###################################################
    def remove_points_inside_level_set(self, level_set_function):
        """Remove points from xyz_coord_P that are inside the level set defined by the level_set_function."""
        mask = level_set_function(*self.xyz_coord_P.T) >= 0
        self.xyz_coord_P = self.xyz_coord_P[mask]

    def smooth_verts(
        self, num_iterations=10, smoothing_factor=0.1, xyz_condition=lambda xyz: True
    ):
        print("Smoothing vertices...")
        Nv = len(self.xyz_coord_V)
        for _ in range(num_iterations):
            print(f"Iteration {_ + 1}/{num_iterations}")
            num_smoothed = 0
            xyz_coord_V = self.xyz_coord_V.copy()
            for v0 in range(Nv):
                num_neighbors = 0
                xyz0 = self.xyz_coord_v(v0)
                if not xyz_condition(xyz0):
                    continue
                num_smoothed += 1
                xyz = np.zeros(3)
                for h in self.generate_H_out_v_clockwise(v0):
                    v = self.v_head_h(h)
                    xyz += self.xyz_coord_v(v)
                    num_neighbors += 1
                if num_neighbors > 0:
                    xyz /= num_neighbors
                    xyz_coord_V[v0] += smoothing_factor * (xyz - xyz0)

            self.xyz_coord_V = xyz_coord_V
            print(f"Number of smoothed vertices: {num_smoothed}")

    # ###################################################
    # ###################################################
    # ###################################################
    # ###################################################
    # ###################################################

    @classmethod
    def from_fib_dumbbell(
        cls, l_max=5, n_points=500, x_squish=0.6, y_squish=0.8, neck_radius=0.2
    ):
        """Create a spherical harmonic surface from a squished Fibonacci sphere. principal components should be aligned with ez, ey, ex for default squish."""
        c = 1 - neck_radius
        raw_point_cloud = fib_sphere(n_points)
        raw_point_cloud[:, 0] *= x_squish * (
            1 - c * np.cos(np.pi * raw_point_cloud[:, 2] / 2)
        )  # Squish along x-axis
        raw_point_cloud[:, 1] *= y_squish * (
            1 - c * np.cos(np.pi * raw_point_cloud[:, 2] / 2)
        )  # Squish along y-axis
        return cls(raw_point_cloud, l_max)
