import numpy as np
from numba import njit, typeof, int64 as NUMBA_INT, float64 as NUMBA_FLOAT
from numba.typed import Dict as NUMBA_DICT, List as NUMBA_LIST
from numba.types import (
    UniTuple as NUMBA_UNITUPLE,
    Tuple as NUMBA_TUPLE,
    Set as NUMBA_SET,
)
from ..mathutils_backend.mesh import (
    tri_vertex_cycles_to_half_edge_samples,
    load_vf_samples_from_ply,
    load_mesh_samples_from_ply,
    write_mesh_samples_to_ply,
)

NUMBA_SIMPLEX0 = NUMBA_UNITUPLE(NUMBA_INT, 1)
NUMBA_SIMPLEX1 = NUMBA_UNITUPLE(NUMBA_INT, 2)
NUMBA_SIMPLEX2 = NUMBA_UNITUPLE(NUMBA_INT, 3)
NUMBA_SIMPLEX3 = NUMBA_UNITUPLE(NUMBA_INT, 4)
NUMBA_DART2 = NUMBA_TUPLE([NUMBA_SIMPLEX0, NUMBA_SIMPLEX1, NUMBA_SIMPLEX2])
NUMBA_DART3 = NUMBA_TUPLE(
    [NUMBA_SIMPLEX0, NUMBA_SIMPLEX1, NUMBA_SIMPLEX2, NUMBA_SIMPLEX3]
)

DART2_0_01_012 = ((0,), (0, 1), (0, 1, 2))
DART2_1_12_012 = ((1,), (1, 2), (0, 1, 2))
DART2_2_02_012 = ((2,), (0, 2), (0, 1, 2))
DART2_1_01_012 = ((1,), (0, 1), (0, 1, 2))
DART2_2_12_012 = ((2,), (1, 2), (0, 1, 2))
DART2_0_02_012 = ((0,), (0, 2), (0, 1, 2))
# NUMBA_DART2_SET = typeof(set([DART2_0_01_012]))
NUMBA_DART2_SET = NUMBA_SET(NUMBA_DART2, True)


@njit
def get_halfedge_index_of_twin(H, h):
    """
    Find the half-edge twin to h in the list of half-edges H.

    Parameters
    ----------
    H : list
        List of half-edges [[v0, v1], ...]
    h : int
        Index of half-edge in H

    Returns
    -------
    h_twin : int
        Index of H[h_twin]=[v1,v0] in H, where H[h]=[v0,v1]. Returns -1 if twin not found.
    """
    hedge_twin = np.flip(H[h])
    ht_arr = np.where((hedge_twin[0] == H[:, 0]) * (hedge_twin[1] == H[:, 1]))[0]
    if ht_arr.size > 0:
        return NUMBA_INT(ht_arr[0])
    return NUMBA_INT(-1)


@njit
def jit_tri_vertex_cycles_to_edge_vertex_cycles(V_cycle_F):
    E = set(
        [
            (0, 0),
        ],
    )
    E.pop()
    for i, j, k in V_cycle_F:
        if i > j:
            i, j = j, i
        if i > k:
            i, k = k, i
        if j > k:
            j, k = k, j
        i, j, k = NUMBA_INT(i), NUMBA_INT(j), NUMBA_INT(k)
        E.update([(i, j), (j, k), (i, k)])
        # E.add((i, j))
    return np.array([[i, j] for i, j in E], dtype=NUMBA_INT)


# def f_samples_to_combinatorialmap(xyz_coord_V, V_cycle_E, V_cycle_F):
#     arrE = np.sort(V_cycle_E, axis=1)
#     arrF = np.sort(V_cycle_F, axis=1)
#     D = set(
#         [
#             (
#                 NUMBA_INT(0),
#                 (NUMBA_INT(0), NUMBA_INT(0)),
#                 (NUMBA_INT(0), NUMBA_INT(0), NUMBA_INT(0)),
#             ),
#         ]
#     )
#     D.pop()
#     for i,j,k


# @njit(inline="always")
# def jit_combinatorialmap_dict_rot(d, D_diff0_D, D_diff1_D, D_diff2_D):
#     return D_diff1_D[D_diff2_D[d]]


def vertex2_orbit(d0, cmap1, cmap2):
    stack = [d0]
    seen = {d0}
    while stack:
        d = stack.pop()
        for cmap in (cmap1, cmap2):
            dn = cmap(d)
            if dn not in seen:
                seen.add(dn)
                stack.append(dn)
    return seen


def edge2_orbit(d0, cmap0, cmap2):
    stack = [d0]
    seen = {d0}
    while stack:
        d = stack.pop()
        for cmap in (cmap0, cmap2):
            dn = cmap(d)
            if dn not in seen:
                seen.add(dn)
                stack.append(dn)
    return seen


def face2_orbit(d0, cmap0, cmap1):
    stack = [d0]
    seen = {d0}
    while stack:
        d = stack.pop()
        for cmap in (cmap0, cmap1):
            dn = cmap(d)
            if dn not in seen:
                seen.add(dn)
                stack.append(dn)
    return seen


def generate_dart_orbit(d0, *cmaps):
    stack = [d0]
    seen = {d0}
    while stack:
        d = stack.pop()
        for cmap in cmaps:
            dn = cmap(d)
            if dn not in seen:
                seen.add(dn)
                stack.append(dn)
                yield dn


def tri_samples_to_combinatorialmap(V_cycle_F):
    NUMBA_INT = int

    V_cycle_F = np.asarray(V_cycle_F, dtype=NUMBA_INT)

    D = set()
    S0 = set()
    S1 = set()
    S2 = set()

    # D_diff0_D = NUMBA_DICT.empty(key_type=NUMBA_DART2, value_type=NUMBA_DART2)
    # D_diff1_D = NUMBA_DICT.empty(key_type=NUMBA_DART2, value_type=NUMBA_DART2)
    # D_diff2_D = NUMBA_DICT.empty(key_type=NUMBA_DART2, value_type=NUMBA_DART2)
    #
    # # darts through k-simplices
    # D_S0 = NUMBA_DICT.empty(key_type=NUMBA_SIMPLEX0, value_type=NUMBA_DART2)
    # D_S1 = NUMBA_DICT.empty(key_type=NUMBA_SIMPLEX1, value_type=NUMBA_DART2)
    # D_S2 = NUMBA_DICT.empty(key_type=NUMBA_SIMPLEX2, value_type=NUMBA_DART2)

    D_diff0_D = dict()
    D_diff1_D = dict()
    D_diff2_D = dict()

    # darts through k-simplices
    D_S0 = dict()
    D_S1 = dict()
    D_S2 = dict()

    # orientedS2 = set()  # 2-chains
    # Define simplices and darts.
    # Define cmap0(v,e,f)=(v',e,f)
    # Define cmap1(v,e,f)=(v,e',f)
    v_max = 0
    for i, j, k in V_cycle_F:
        # i, j, k = NUMBA_INT(i), NUMBA_INT(j), NUMBA_INT(k)
        # orientedS2.add((i, j, k))
        if i > j:
            i, j = j, i
        if i > k:
            i, k = k, i
        if j > k:
            j, k = k, j
        if k > v_max:
            v_max = k

        s_i, s_j, s_k = (i,), (j,), (k,)
        s_ij = (i, j)
        s_jk = (j, k)
        s_ik = (i, k)
        s_ijk = (i, j, k)

        i_ij_ijk = ((i,), (i, j), (i, j, k))
        j_jk_ijk = ((j,), (j, k), (i, j, k))
        k_ik_ijk = ((k,), (i, k), (i, j, k))

        j_ij_ijk = ((j,), (i, j), (i, j, k))
        k_jk_ijk = ((k,), (j, k), (i, j, k))
        i_ik_ijk = ((i,), (i, k), (i, j, k))

        S0.update([s_i, s_j, s_k])
        S1.update([s_ij, s_jk, s_ik])
        S2.add(s_ijk)
        D.update(
            [
                i_ij_ijk,
                j_jk_ijk,
                k_ik_ijk,
                j_ij_ijk,
                k_jk_ijk,
                i_ik_ijk,
            ],
        )

        D_S0[s_i] = i_ij_ijk
        D_S0[s_j] = j_jk_ijk
        D_S0[s_k] = k_ik_ijk

        D_S1[s_ij] = i_ij_ijk
        D_S1[s_jk] = j_jk_ijk
        D_S1[s_ik] = i_ik_ijk

        D_S2[s_ijk] = i_ij_ijk

        D_diff0_D[i_ij_ijk] = j_ij_ijk
        D_diff0_D[j_jk_ijk] = k_jk_ijk
        D_diff0_D[k_ik_ijk] = i_ik_ijk
        D_diff0_D[j_ij_ijk] = i_ij_ijk
        D_diff0_D[k_jk_ijk] = j_jk_ijk
        D_diff0_D[i_ik_ijk] = k_ik_ijk

        D_diff1_D[i_ij_ijk] = i_ik_ijk
        D_diff1_D[j_jk_ijk] = j_ij_ijk
        D_diff1_D[k_ik_ijk] = k_jk_ijk
        D_diff1_D[j_ij_ijk] = j_jk_ijk
        D_diff1_D[k_jk_ijk] = k_ik_ijk
        D_diff1_D[i_ik_ijk] = i_ij_ijk

    Dbdry = set()
    needdiff2D = D.copy()
    # Define d'=D_diff2_D[d] with same v'=v, e'=e, and f'!=f.
    # Set D_diff2_D[d]=d for boundary darts.
    while needdiff2D:
        d = needdiff2D.pop()
        (v, e, f) = d
        D_diff2_D[d] = d
        for dd in needdiff2D:
            (vv, ee, ff) = d
            if v == vv and e == ee:  # never d==dd since d already popped
                D_diff2_D[d] = dd
                D_diff2_D[dd] = d
                needdiff2D.remove(dd)
                break
        if d == dd:
            Dbdry.add(d)

    def twin(d):
        return D_diff0_D[D_diff2_D[d]]

    def next(d):
        return D_diff1_D[D_diff0_D[d]]

    # def prev(d):
    #     return D_diff0_D[D_diff1_D[d]]

    def rot_cw(d):
        return D_diff1_D[D_diff2_D[d]]

    def rot_ccw(d):
        return D_diff2_D[D_diff1_D[d]]

    # print(f"{len(D_diff2_D)=}") # this f-string breaks numba
    #
    Dplus = set()
    Dminus = set()
    need2visitD = D.copy()
    d_plus = need2visitD.pop()
    while need2visitD:
        while True:
            d_minus = D_diff0_D[d_plus]
            need2visitD.discard(d_plus)
            need2visitD.discard(d_minus)
            if d_plus in Dplus:
                d_plus = rot_ccw(d_plus)
                d_plus = next(d_plus)
                break
            Dplus.add(d_plus)
            Dminus.add(d_minus)
            d_plus = rot_cw(d_plus)

    print(f"{len(D)=}, {len(Dplus)=}, {len(Dminus)=}")
    bad_next = 0
    bad_twin = 0
    for d in Dplus:
        v, e, f = d
        next_d = next(d)
        twin_d = twin(d)
        if next_d not in Dplus:
            bad_next += 1
        elif next_d not in Dminus:
            raise ValueError
        if twin_d not in Dplus:
            bad_twin += 1
        elif twin_d not in Dminus:
            raise ValueError

    num_darts = len(D)
    num_verts = len(S0)
    num_edges = len(S1)
    num_faces = len(S2)
    num_half_edges = len(Dplus)
    num_boundaries = len(Dbdry)

    i, j, k = V_cycle_F[0]
    positive = True
    if i > j:
        i, j = j, i
        positive = not positive
    if i > k:
        i, k = k, i
        positive = not positive
    if j > k:
        j, k = k, j
        positive = not positive
    i_ij_ijk = ((i,), (i, j), (i, j, k))
    if i_ij_ijk in Dminus:
        Dplus, Dminus = Dminus, Dplus
    elif i_ij_ijk not in Dplus:
        raise ValueError

    enumeration_Dplus = {d: _ for _, d in enumerate(Dplus)}
    # enumeration_S0 = {s: _ for _, s in enumerate(S0)}
    enumeration_S1 = {s: _ for _, s in enumerate(S1)}
    enumeration_S2 = {s: _ for _, s in enumerate(S2)}

    v_origin_H = np.empty(num_half_edges, dtype=NUMBA_INT)
    e_undirected_H = np.empty(num_half_edges, dtype=NUMBA_INT)
    f_left_H = np.empty(num_half_edges, dtype=NUMBA_INT)

    h_next_H = np.empty(num_half_edges, dtype=NUMBA_INT)
    h_twin_H = np.empty(num_half_edges, dtype=NUMBA_INT)

    h_out_V = np.empty(num_verts, dtype=NUMBA_INT)
    h_directed_E = np.empty(num_edges, dtype=NUMBA_INT)
    h_right_F = np.empty(num_faces, dtype=NUMBA_INT)
    h_negative_B = np.empty(num_boundaries, dtype=NUMBA_INT)

    for d, h in enumeration_Dplus.items():
        s_v, s_e, s_f = d
        v = s_v[0]
        e = enumeration_S1[s_e]
        f = enumeration_S2[s_f]

        v_origin_H[h] = v
        e_undirected_H[h] = enumeration_S1[s_e]
        f_left_H[h] = enumeration_S2[s_f]

        h_next_H[h] = enumeration_Dplus[next(d)]
        h_twin_H[h] = enumeration_Dplus[twin(d)]

        h_out_V[v] = enumeration_Dplus[d]
        h_directed_E[e] = enumeration_Dplus[d]
        h_right_F[f] = enumeration_Dplus[d]

    V_cycle_E = np.empty((len(S1), 2), dtype=NUMBA_INT)
    for _, e in enumeration_S1.items():
        # h = h_directed_E[e]
        V_cycle_E[e] = (
            v_origin_H[h_directed_E[e]],
            v_origin_H[h_next_H[h_directed_E[e]]],
        )
    V_cycle_F_ = np.empty((len(S2), 3), dtype=NUMBA_INT)
    for ijk, f in enumeration_S2.items():
        V_cycle_F_[f] = (
            v_origin_H[h_right_F[f]],
            v_origin_H[h_next_H[h_right_F[f]]],
            v_origin_H[h_next_H[h_next_H[h_right_F[f]]]],
        )

    cmaps = {
        "D_diff0_D": D_diff0_D,
        "D_diff1_D": D_diff1_D,
        "D_diff2_D": D_diff2_D,
    }
    he_samples = {
        "v_origin_H": v_origin_H,
        "e_undirected_H": e_undirected_H,
        "f_left_H": f_left_H,
        "h_next_H": h_next_H,
        "h_twin_H": h_twin_H,
        "h_out_V": h_out_V,
        "h_directed_E": h_directed_E,
        "h_right_F": h_right_F,
        "h_negative_B": h_negative_B,
    }
    v_cycles = {
        "V_cycle_E": V_cycle_E,
        "V_cycle_F": V_cycle_F_,
    }
    return (
        cmaps,
        D_S0,
        D_S1,
        D_S2,
        D,
        Dplus,
        Dminus,
        S0,
        S1,
        S2,
        enumeration_Dplus,
        he_samples,
        v_cycles,
    )


@njit
def jit_tri_samples_to_combinatorialmap(V_cycle_F):
    V_cycle_F = np.asarray(V_cycle_F, dtype=NUMBA_INT)

    D = set()
    S0 = set()
    S1 = set()
    S2 = set()

    D_diff0_D = NUMBA_DICT.empty(key_type=NUMBA_DART2, value_type=NUMBA_DART2)
    D_diff1_D = NUMBA_DICT.empty(key_type=NUMBA_DART2, value_type=NUMBA_DART2)
    D_diff2_D = NUMBA_DICT.empty(key_type=NUMBA_DART2, value_type=NUMBA_DART2)

    # darts through k-simplices
    D_S0 = NUMBA_DICT.empty(key_type=NUMBA_SIMPLEX0, value_type=NUMBA_DART2)
    D_S1 = NUMBA_DICT.empty(key_type=NUMBA_SIMPLEX1, value_type=NUMBA_DART2)
    D_S2 = NUMBA_DICT.empty(key_type=NUMBA_SIMPLEX2, value_type=NUMBA_DART2)

    # orientedS2 = set()  # 2-chains
    # Define simplices and darts.
    # Define cmap0(v,e,f)=(v',e,f)
    # Define cmap1(v,e,f)=(v,e',f)
    v_max = 0
    for i, j, k in V_cycle_F:
        # i, j, k = NUMBA_INT(i), NUMBA_INT(j), NUMBA_INT(k)
        # orientedS2.add((i, j, k))
        if i > j:
            i, j = j, i
        if i > k:
            i, k = k, i
        if j > k:
            j, k = k, j
        if k > v_max:
            v_max = k

        s_i, s_j, s_k = (i,), (j,), (k,)
        s_ij = (i, j)
        s_jk = (j, k)
        s_ik = (i, k)
        s_ijk = (i, j, k)

        i_ij_ijk = ((i,), (i, j), (i, j, k))
        j_jk_ijk = ((j,), (j, k), (i, j, k))
        k_ik_ijk = ((k,), (i, k), (i, j, k))

        j_ij_ijk = ((j,), (i, j), (i, j, k))
        k_jk_ijk = ((k,), (j, k), (i, j, k))
        i_ik_ijk = ((i,), (i, k), (i, j, k))

        S0.update([s_i, s_j, s_k])
        S1.update([s_ij, s_jk, s_ik])
        S2.add(s_ijk)
        D.update(
            [
                i_ij_ijk,
                j_jk_ijk,
                k_ik_ijk,
                j_ij_ijk,
                k_jk_ijk,
                i_ik_ijk,
            ],
        )

        D_S0[s_i] = i_ij_ijk
        D_S0[s_j] = j_jk_ijk
        D_S0[s_k] = k_ik_ijk

        D_S1[s_ij] = i_ij_ijk
        D_S1[s_jk] = j_jk_ijk
        D_S1[s_ik] = i_ik_ijk

        D_S2[s_ijk] = i_ij_ijk

        D_diff0_D[i_ij_ijk] = j_ij_ijk
        D_diff0_D[j_jk_ijk] = k_jk_ijk
        D_diff0_D[k_ik_ijk] = i_ik_ijk
        D_diff0_D[j_ij_ijk] = i_ij_ijk
        D_diff0_D[k_jk_ijk] = j_jk_ijk
        D_diff0_D[i_ik_ijk] = k_ik_ijk

        D_diff1_D[i_ij_ijk] = i_ik_ijk
        D_diff1_D[j_jk_ijk] = j_ij_ijk
        D_diff1_D[k_ik_ijk] = k_jk_ijk
        D_diff1_D[j_ij_ijk] = j_jk_ijk
        D_diff1_D[k_jk_ijk] = k_ik_ijk
        D_diff1_D[i_ik_ijk] = i_ij_ijk

    Dbdry = set()
    needdiff2D = D.copy()
    # Define d'=D_diff2_D[d] with same v'=v, e'=e, and f'!=f.
    # Set D_diff2_D[d]=d for boundary darts.
    while needdiff2D:
        d = needdiff2D.pop()
        (v, e, f) = d
        D_diff2_D[d] = d
        for dd in needdiff2D:
            (vv, ee, ff) = d
            if v == vv and e == ee:  # never d==dd since d already popped
                D_diff2_D[d] = dd
                D_diff2_D[dd] = d
                needdiff2D.remove(dd)
                break
        if d == dd:
            Dbdry.add(d)

    def twin(d):
        return D_diff0_D[D_diff2_D[d]]

    def next(d):
        return D_diff1_D[D_diff0_D[d]]

    # def prev(d):
    #     return D_diff0_D[D_diff1_D[d]]

    def rot_cw(d):
        return D_diff1_D[D_diff2_D[d]]

    def rot_ccw(d):
        return D_diff2_D[D_diff1_D[d]]

    # print(f"{len(D_diff2_D)=}") # this f-string breaks numba
    #
    Dplus = set()
    Dminus = set()
    need2visitD = D.copy()
    d_plus = need2visitD.pop()
    while need2visitD:
        while True:
            d_minus = D_diff0_D[d_plus]
            need2visitD.discard(d_plus)
            need2visitD.discard(d_minus)
            if d_plus in Dplus:
                d_plus = rot_ccw(d_plus)
                d_plus = next(d_plus)
                break
            Dplus.add(d_plus)
            Dminus.add(d_minus)
            d_plus = rot_cw(d_plus)

    bad_next = 0
    bad_twin = 0
    for d in Dplus:
        v, e, f = d
        next_d = next(d)
        twin_d = twin(d)
        if next_d not in Dplus:
            bad_next += 1
        elif next_d not in Dminus:
            raise ValueError
        if twin_d not in Dplus:
            bad_twin += 1
        elif twin_d not in Dminus:
            raise ValueError

    num_darts = len(D)
    num_verts = len(S0)
    num_edges = len(S1)
    num_faces = len(S2)
    num_half_edges = len(Dplus)
    num_boundaries = len(Dbdry)

    i, j, k = V_cycle_F[0]
    positive = True
    if i > j:
        i, j = j, i
        positive = not positive
    if i > k:
        i, k = k, i
        positive = not positive
    if j > k:
        j, k = k, j
        positive = not positive
    i_ij_ijk = ((i,), (i, j), (i, j, k))
    if i_ij_ijk in Dminus:
        Dplus, Dminus = Dminus, Dplus
    elif i_ij_ijk not in Dplus:
        raise ValueError

    enumeration_Dplus = {d: _ for _, d in enumerate(Dplus)}
    # enumeration_S0 = {s: _ for _, s in enumerate(S0)}
    enumeration_S1 = {s: _ for _, s in enumerate(S1)}
    enumeration_S2 = {s: _ for _, s in enumerate(S2)}

    v_origin_H = np.empty(num_half_edges, dtype=NUMBA_INT)
    e_undirected_H = np.empty(num_half_edges, dtype=NUMBA_INT)
    f_left_H = np.empty(num_half_edges, dtype=NUMBA_INT)

    h_next_H = np.empty(num_half_edges, dtype=NUMBA_INT)
    h_twin_H = np.empty(num_half_edges, dtype=NUMBA_INT)

    h_out_V = np.empty(num_verts, dtype=NUMBA_INT)
    h_directed_E = np.empty(num_edges, dtype=NUMBA_INT)
    h_right_F = np.empty(num_faces, dtype=NUMBA_INT)
    h_negative_B = np.empty(num_boundaries, dtype=NUMBA_INT)

    for d, h in enumeration_Dplus.items():
        s_v, s_e, s_f = d
        v = s_v[0]
        e = enumeration_S1[s_e]
        f = enumeration_S2[s_f]

        v_origin_H[h] = v
        e_undirected_H[h] = enumeration_S1[s_e]
        f_left_H[h] = enumeration_S2[s_f]

        h_next_H[h] = enumeration_Dplus[next(d)]
        h_twin_H[h] = enumeration_Dplus[twin(d)]

        h_out_V[v] = enumeration_Dplus[d]
        h_directed_E[e] = enumeration_Dplus[d]
        h_right_F[f] = enumeration_Dplus[d]

    V_cycle_E = np.empty((len(S1), 2), dtype=NUMBA_INT)
    for _, e in enumeration_S1.items():
        # h = h_directed_E[e]
        V_cycle_E[e] = (
            v_origin_H[h_directed_E[e]],
            v_origin_H[h_next_H[h_directed_E[e]]],
        )
    V_cycle_F_ = np.empty((len(S2), 3), dtype=NUMBA_INT)
    for ijk, f in enumeration_S2.items():
        V_cycle_F_[f] = (
            v_origin_H[h_right_F[f]],
            v_origin_H[h_next_H[h_right_F[f]]],
            v_origin_H[h_next_H[h_next_H[h_right_F[f]]]],
        )

    cmaps = {
        "D_diff0_D": D_diff0_D,
        "D_diff1_D": D_diff1_D,
        "D_diff2_D": D_diff2_D,
    }
    he_samples = {
        "v_origin_H": v_origin_H,
        "e_undirected_H": e_undirected_H,
        "f_left_H": f_left_H,
        "h_next_H": h_next_H,
        "h_twin_H": h_twin_H,
        "h_out_V": h_out_V,
        "h_directed_E": h_directed_E,
        "h_right_F": h_right_F,
        "h_negative_B": h_negative_B,
    }
    v_cycles = {
        "V_cycle_E": V_cycle_E,
        "V_cycle_F": V_cycle_F_,
    }
    return (
        cmaps,
        D_S0,
        D_S1,
        D_S2,
        D,
        Dplus,
        Dminus,
        S0,
        S1,
        S2,
        enumeration_Dplus,
        he_samples,
        v_cycles,
    )


@njit
def vf_samples_to_he_samples(xyz_coord_V, V_cycle_F):
    Nfaces = len(V_cycle_F)
    Nvertices = len(xyz_coord_V)
    _Nhedges = 3 * Nfaces * 2
    _H = np.zeros((_Nhedges, 2), dtype=NUMBA_INT)
    h_out_V = -np.ones(Nvertices, dtype=NUMBA_INT)
    _v_origin_H = np.zeros(_Nhedges, dtype=NUMBA_INT)
    _h_next_H = -np.ones(_Nhedges, dtype=NUMBA_INT)
    _f_left_H = np.zeros(_Nhedges, dtype=NUMBA_INT)
    h_right_F = np.zeros(Nfaces, dtype=NUMBA_INT)

    # h_count = 0
    for f in range(Nfaces):
        h_right_F[f] = 3 * f
        for i in range(3):
            h = 3 * f + i
            h_next = 3 * f + (i + 1) % 3
            v0 = V_cycle_F[f, i]
            v1 = V_cycle_F[f, (i + 1) % 3]
            _H[h] = [v0, v1]
            _v_origin_H[h] = v0
            _f_left_H[h] = f
            _h_next_H[h] = h_next
            if h_out_V[v0] == -1:
                h_out_V[v0] = h
    h_count = 3 * Nfaces
    need_twins = set([NUMBA_INT(_) for _ in range(h_count)])
    need_next = set([NUMBA_INT(0)])
    need_next.pop()
    _h_twin_H = NUMBA_INT(-2) * np.ones(_Nhedges, dtype=NUMBA_INT)  # -2 means not set
    while need_twins:
        h = need_twins.pop()
        if _h_twin_H[h] == -2:  # if twin not set
            h_twin = get_halfedge_index_of_twin(_H, h)  # returns -1 if twin not found
            if h_twin == -1:  # if twin not found
                h_twin = NUMBA_INT(h_count)
                h_count += 1
                v0, v1 = _H[h]
                _H[h_twin] = [v1, v0]
                _v_origin_H[h_twin] = v1
                need_next.add(h_twin)
                _h_twin_H[h] = h_twin
                _h_twin_H[h_twin] = h
                _f_left_H[h_twin] = -1
            else:
                _h_twin_H[h], _h_twin_H[h_twin] = h_twin, h
                need_twins.remove(NUMBA_INT(h_twin))

    Nhedges = h_count
    # H = _H[:Nhedges]
    v_origin_H = _v_origin_H[:Nhedges]
    h_next_H = _h_next_H[:Nhedges]
    f_left_H = _f_left_H[:Nhedges]
    h_twin_H = _h_twin_H[:Nhedges]
    while need_next:
        h = need_next.pop()
        h_next = h_twin_H[h]
        # rotate ccw around origin of twin until we find nex h on boundary
        while f_left_H[h_next] != -1:
            h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
        h_next_H[h] = h_next

    # find and enumerate boundaries -1,-2,...
    H_need2visit = set([NUMBA_INT(h) for h in range(Nhedges) if f_left_H[h] == -1])
    _h_comp_B = []
    while H_need2visit:
        b = len(_h_comp_B)
        h_start = H_need2visit.pop()
        f_left_H[h_start] = -(b + 1)
        h = NUMBA_INT(h_next_H[h_start])
        _h_comp_B.append(h)
        while h != h_start:
            H_need2visit.remove(h)
            f_left_H[h] = -(b + 1)
            h = h_next_H[h]
    h_negative_B = np.array(_h_comp_B, dtype=NUMBA_INT)
    return (
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_right_F,
        h_negative_B,
    )


@njit
def vertex_pair_key(v0, v1):
    return min(v0, v1) * 1000000 + max(v0, v1)


@njit
def jit_refine_icososphere(Vm1, Fm1, r):
    # Nfaces = len(Fm1)
    # F = np.zeros((4*Nfaces, 3), dtype=int)
    F = []
    V = [xyz for xyz in Vm1]
    v_midpt_vv = NUMBA_DICT.empty(key_type=NUMBA_INT, value_type=NUMBA_INT)
    for tri in Fm1:
        v0, v1, v2 = tri
        key01 = vertex_pair_key(v0, v1)
        key12 = vertex_pair_key(v1, v2)
        key20 = vertex_pair_key(v2, v0)
        v01 = v_midpt_vv.get(key01, NUMBA_INT(-1))
        v12 = v_midpt_vv.get(key12, NUMBA_INT(-1))
        v20 = v_midpt_vv.get(key20, NUMBA_INT(-1))
        if v01 == NUMBA_INT(-1):
            v01 = NUMBA_INT(len(V))
            xyz01 = (V[v0] + V[v1]) / 2
            xyz01 *= r / np.linalg.norm(xyz01)
            V.append(xyz01)
            v_midpt_vv[key01] = v01
        if v12 == NUMBA_INT(-1):
            v12 = NUMBA_INT(len(V))
            xyz12 = (V[v1] + V[v2]) / 2
            xyz12 *= r / np.linalg.norm(xyz12)
            V.append(xyz12)
            v_midpt_vv[key12] = v12
        if v20 == NUMBA_INT(-1):
            v20 = NUMBA_INT(len(V))
            xyz20 = (V[v2] + V[v0]) / 2
            xyz20 *= r / np.linalg.norm(xyz20)
            V.append(xyz20)
            v_midpt_vv[key20] = v20
        F.append([v0, v01, v20])
        F.append([v01, v1, v12])
        F.append([v20, v12, v2])
        F.append([v01, v12, v20])

    Nv = len(V)
    Nf = len(F)
    xyz_coord_V = np.zeros((Nv, 3), dtype=NUMBA_FLOAT)
    V_cycle_F = np.zeros((Nf, 3), dtype=NUMBA_INT)
    for v in range(Nv):
        xyz_coord_V[v] = V[v]
    for f in range(Nf):
        V_cycle_F[f] = F[f]
    return xyz_coord_V, V_cycle_F


class HalfEdgeMesh:
    """
    Array-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.ndarray of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as ndarrays of vertex/half-edge/face labels. Each data array has a name of the form "a_description_Q", where "a" denotes the type of object associated with the elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "Q" denotes the type of object associated with the indices ("V" for vertex, "H" for half-edge, "F" for face, or "B" for boundary), and "description" is a description of information represented by the data. For example, "v_origin_H_" is an array of vertices at the origin of each half-edge. The i-th element of data array "a_description_Q" can be accessed using the "a_description_q(i)" method.

    Properties
    ----------
    xyz_coord_V : ndarray[:, :] of float
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of int
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of int
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of int
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of int
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of int
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_right_F : ndarray[:] of int
        h_right_F[k] = some half-edge on the boudary of face k.
    h_negative_B : ndarray[:] of int
        h_negative_B[n] = half-edge to the right of boundary n.

    Initialization
    ---------------
    The HalfEdgeMesh class can be initialized in several ways:
    - Directly from half-edge mesh data arrays:
        HalfEdgeMesh(xyz_coord_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_right_F,
                     h_negative_B)
    - From an npz file containing data arrays:
        HalfEdgeMesh.load_he_samples(npz_path)
    - From an array of vertex positions and an array of face vertices:
        HalfEdgeMesh.from_vf_samples(xyz_coord_V, V_cycle_F)
    """

    def __init__(
        self,
        xyz_coord_V=None,
        h_out_V=None,
        v_origin_H=None,
        h_next_H=None,
        h_twin_H=None,
        f_left_H=None,
        h_right_F=None,
        h_negative_B=None,
        V_cycle_E=None,
        V_cycle_F=None,
        *args,
        **kwargs,
    ):
        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.f_left_H = f_left_H
        self.h_right_F = h_right_F
        self.h_negative_B = h_negative_B

        defaults = self.default_samples()  # | self.default_ragged_samples()
        self.default_sample_keys = list(defaults.keys())

        for key, val in defaults.items():
            if getattr(self, key, None) is None:
                self.__setattr__(key, val)

    @classmethod
    def default_samples(cls):
        return {
            "xyz_coord_V": np.empty((0, 3), dtype=float),
            "h_out_V": np.empty((0,), dtype=int),
            "V_cycle_E": np.empty((0, 2), dtype=int),
            "h_directed_E": np.empty((0,), dtype=int),
            "V_cycle_F": np.empty((0, 3), dtype=int),
            "h_right_F": np.empty((0,), dtype=int),
            "v_origin_H": np.empty((0,), dtype=int),
            "e_undirected_H": np.empty((0,), dtype=int),
            "f_left_H": np.empty((0,), dtype=int),
            "h_next_H": np.empty((0,), dtype=int),
            "h_twin_H": np.empty((0,), dtype=int),
            "h_negative_B": np.empty((0,), dtype=int),
        }

    @classmethod
    def default_ragged_samples(cls):
        return {
            "V_cycle_B": [],
            "H_cycle_B": [],
        }

    #######################################################
    # Initilization methods
    ######################################################
    @classmethod
    def init_empty(cls, *args, **kwargs):
        return cls(
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            *args,
            **kwargs,
        )

    @classmethod
    def init_icososphere(cls, num_refinements=0, radius=1.0, compute_he_stuff=False):
        phi = (1.0 + np.sqrt(5.0)) * 0.5
        a = 1.0
        b = 1.0 / phi

        # num_vertices = 12
        # num_faces = 20
        # xyz_coord_V = np.zeros(num_vertices, 3, dtype=float)
        # V_cycle_F = np.zeros(num_faces, 3, dtype=int)
        xyz_coord_V = np.array(
            [
                [0.0, b, -a],
                [b, a, 0.0],
                [-b, a, 0.0],
                [0.0, b, a],
                [0.0, -b, a],
                [-a, 0.0, b],
                [0.0, -b, -a],
                [a, 0.0, -b],
                [a, 0.0, b],
                [-a, 0.0, -b],
                [b, -a, 0.0],
                [-b, -a, 0.0],
            ],
            dtype=float,
        )

        rad = np.sqrt(a * a + b * b)
        xyz_coord_V /= rad
        V_cycle_F = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
                [5, 4, 3],
                [4, 8, 3],
                [7, 6, 0],
                [6, 9, 0],
                [11, 10, 4],
                [10, 11, 6],
                [9, 5, 2],
                [5, 9, 11],
                [8, 7, 1],
                [7, 8, 10],
                [2, 5, 3],
                [8, 1, 3],
                [9, 2, 0],
                [1, 7, 0],
                [11, 9, 6],
                [7, 10, 6],
                [5, 11, 4],
                [10, 8, 4],
            ],
            dtype=int,
        )
        xyz_coord_V *= radius
        for _ in range(num_refinements):
            xyz_coord_V, V_cycle_F = jit_refine_icososphere(
                xyz_coord_V, V_cycle_F, radius
            )
        return cls.from_vf_samples(
            xyz_coord_V, V_cycle_F, compute_he_stuff=compute_he_stuff
        )

    @classmethod
    def from_samples(cls, **samples_dict):
        he_keys = [
            "v_origin_H",
            "e_undirected_H",
            "f_left_H",
            "h_next_H",
            "h_twin_H",
            "h_out_V",
            "h_directed_E",
            "h_right_F",
            "h_negative_B",
        ]
        he_keys += [
            "xyz_coord_V",
        ]
        he_keys += [
            "V_cycle_E",
            "V_cycle_F",
        ]
        # he_samples = {key: samples_dict.get(key, np.zeros(0, dtype=int)) for key in he_keys}
        m = cls.init_empty()
        for name, value in samples_dict.items():
            m.__setattr__(name, value)
        return m

    @classmethod
    def from_vf_samples(
        cls, xyz_coord_V, V_cycle_F, *args, compute_he_stuff=True, **kwargs
    ):
        """
        Initialize a half-edge mesh from vertex/face data.

        Parameters:
        ----------
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i
        V_cycle_F : list of lists of integers
            V_cycle_F[j] = [v0, v1, v2] = vertices in face j.

        Returns:
        -------
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with the given vertices and faces.
        """
        if compute_he_stuff:
            m = cls(
                *vf_samples_to_he_samples(xyz_coord_V, V_cycle_F),
                # *args,
                # **kwargs,
            )
            m.V_cycle_F = V_cycle_F
        else:
            m = cls.init_empty()
            m.xyz_coord_V = xyz_coord_V
            m.V_cycle_F = V_cycle_F
        return m

    @classmethod
    def load_vf_ply(cls, ply_path, *args, compute_he_stuff=True, **kwargs):
        """
        Initialize a half-edge mesh from vertex/face data.

        Parameters:
        ----------
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i
        V_cycle_F : list of lists of integers
            V_cycle_F[j] = [v0, v1, v2] = vertices in face j.

        Returns:
        -------
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with the given vertices and faces.
        """
        xyz_coord_V, V_cycle_F = load_vf_samples_from_ply(ply_path)
        if compute_he_stuff:
            m = cls(
                *vf_samples_to_he_samples(xyz_coord_V, V_cycle_F),
                # *args,
                # **kwargs,
            )
            m.V_cycle_F = V_cycle_F
        else:
            m = cls.init_empty()
            m.xyz_coord_V = xyz_coord_V
            m.V_cycle_F = V_cycle_F
        return m

    @classmethod
    def load_ply(cls, ply_path, *args, compute_he_stuff=True, **kwargs):
        """
        Initialize a half-edge mesh from vertex/face data.
        """
        return cls.from_samples(**load_mesh_samples_from_ply(ply_path))

    def save_ply(self, ply_path):
        """
        Write vertex/face data to a PLY file.

        Args:
            ply_path (str): path to save file
        """
        # mesh_samples = {
        #     "xyz_coord_V": self.xyz_coord_V,
        #     "h_out_V": self.h_out_V,
        #     "V_cycle_E": self.V_cycle_E,
        #     "h_directed_E": self.h_directed_E,
        #     "V_cycle_F": self.V_cycle_F,
        #     "h_right_F": self.h_right_F,
        #     "v_origin_H": self.v_origin_H,
        #     "e_undirected_H": self.e_undirected_H,
        #     "f_left_H": self.f_left_H,
        #     "h_twin_H": self.h_twin_H,
        #     "h_next_H": self.h_next_H,
        # }
        write_mesh_samples_to_ply(
            self.to_mesh_samples(),
            ply_path,
        )

    @classmethod
    def load_he_samples(cls, npz_path, *args, **kwargs):
        """Initialize a half-edge mesh from npz file containing data arrays."""
        data = np.load(npz_path)
        return cls(
            data["xyz_coord_V"],
            data["h_out_V"],
            data["v_origin_H"],
            data["h_next_H"],
            data["h_twin_H"],
            data["f_left_H"],
            data["h_right_F"],
            data["h_negative_B"],
            *args,
            **kwargs,
        )

    def save_he_samples(self, path):
        """
        Save data arrays to npz file

        Args:
            path (str): path to save file
        """
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_right_F,
            h_negative_B,
        ) = self.he_samples
        np.savez(
            path,
            xyz_coord_V=xyz_coord_V,
            h_out_V=h_out_V,
            v_origin_H=v_origin_H,
            h_next_H=h_next_H,
            h_twin_H=h_twin_H,
            f_left_H=f_left_H,
            h_right_F=h_right_F,
            h_negative_B=h_negative_B,
        )

    #######################################################
    # Fundamental accessors and properties ###############
    #######################################################
    @property
    def xyz_coord_V(self):
        return self.xyz_coord_V_

    @xyz_coord_V.setter
    def xyz_coord_V(self, value):
        self.xyz_coord_V_ = np.array(value, dtype=float)

    @property
    def h_out_V(self):
        return self.h_out_V_

    @h_out_V.setter
    def h_out_V(self, value):
        self.h_out_V_ = np.array(value, dtype=int)

    @property
    def v_origin_H(self):
        return self.v_origin_H_

    @v_origin_H.setter
    def v_origin_H(self, value):
        self.v_origin_H_ = np.array(value, dtype=int)

    @property
    def h_next_H(self):
        return self.h_next_H_

    @h_next_H.setter
    def h_next_H(self, value):
        self.h_next_H_ = np.array(value, dtype=int)

    @property
    def h_twin_H(self):
        return self.h_twin_H_

    @h_twin_H.setter
    def h_twin_H(self, value):
        self.h_twin_H_ = np.array(value, dtype=int)

    @property
    def f_left_H(self):
        return self.f_left_H_

    @f_left_H.setter
    def f_left_H(self, value):
        self.f_left_H_ = np.array(value, dtype=int)

    @property
    def h_right_F(self):
        return self.h_right_F_

    @h_right_F.setter
    def h_right_F(self, value):
        self.h_right_F_ = np.array(value, dtype=int)

    @property
    def h_negative_B(self):
        return self.h_negative_B_

    @h_negative_B.setter
    def h_negative_B(self, value):
        self.h_negative_B_ = np.array(value, dtype=int)

    @property
    def e_undirected_H(self):
        return self.e_undirected_H_

    @e_undirected_H.setter
    def e_undirected_H(self, value):
        self.e_undirected_H_ = np.array(value, dtype=int)

    @property
    def h_directed_E(self):
        return self.h_directed_E_

    @h_directed_E.setter
    def h_directed_E(self, value):
        self.h_directed_E_ = np.array(value, dtype=int)

    @property
    def V_cycle_E(self):
        if len(self.V_cycle_E_) > 0:
            return self.V_cycle_E_
        self.V_cycle_E = jit_tri_vertex_cycles_to_edge_vertex_cycles(self.V_cycle_F)
        return self.V_cycle_E_

    @V_cycle_E.setter
    def V_cycle_E(self, value):
        self.V_cycle_E_ = np.array(value, dtype=int)

    @property
    def V_cycle_F(self):
        if len(self.V_cycle_F_) > 0:
            return self.V_cycle_F_
        self.V_cycle_F = np.array(
            [
                self.v_origin_H[self.h_right_F],
                self.v_origin_H[self.h_next_H[self.h_right_F]],
                self.v_origin_H[self.h_next_H[self.h_next_H[self.h_right_F]]],
            ]
        ).T
        return self.V_cycle_F_

    @V_cycle_F.setter
    def V_cycle_F(self, value):
        self.V_cycle_F_ = np.array(value, dtype=int)

    @property
    def num_vertices(self):
        return len(self.xyz_coord_V)

    @property
    def num_edges(self):
        return len(self.v_origin_H_) // 2

    @property
    def num_half_edges(self):
        return len(self.v_origin_H_)

    @property
    def num_faces(self):
        return len(self.h_right_F_)

    @property
    def euler_characteristic(self):
        return self.num_vertices - self.num_edges + self.num_faces

    @property
    def num_boundaries(self):
        return len(self.h_negative_B_)

    @property
    def genus(self):
        return 1 - (self.euler_characteristic + self.num_boundaries) // 2

    @property
    def mesh_samples(self):
        # return {
        #     "xyz_coord_V": self.xyz_coord_V,
        #     "h_out_V": self.h_out_V,
        #     "v_origin_H": self.v_origin_H,
        #     "h_next_H": self.h_next_H,
        #     "h_twin_H": self.h_twin_H,
        #     "f_left_H": self.f_left_H,
        #     "h_right_F": self.h_right_F,
        #     "h_negative_B": self.h_negative_B,
        # }
        return {k: self.__getattribute__(k) for k in self.default_sample_keys}

    def to_mesh_samples(self):
        return {k: self.__getattribute__(k) for k in self.default_sample_keys}

    #######################################################
    # Combinatorial maps #################################
    #######################################################
    def xyz_coord_v(self, v):
        """get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self.xyz_coord_V[v]

    def h_out_v(self, v):
        """
        get index of an outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        return self.h_out_V_[v]

    def h_right_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        return self.h_right_F_[f]

    def v_origin_h(self, h):
        """get index of the vertex at the origin of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self.v_origin_H_[h]

    def f_left_h(self, h):
        """get index of the face to the left of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: face index
        """
        return self.f_left_H_[h]

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face cycle

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self.h_next_H_[h]

    def h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self.h_twin_H_[h]

    def h_right_b(self, b):
        """get index of a half-edge contained in boundary b

        Args:
            b (int): boundary index

        Returns:
            int: half-edge index
        """
        if b < 0:
            return self.h_negative_B_[-(b + 1)]
        return self.h_negative_B_[b]

    # Derived combinatorial maps
    def h_in_v(self, v):
        """get index of an incoming half-edge incident on vertex v"""
        return self.h_twin_h(self.h_out_v(v))

    def v_head_h(self, h):
        """get index of the vertex at the head of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self.v_origin_h(self.h_twin_h(h))

    def h_rotcw_h(self, h):
        return self.h_next_h(self.h_twin_h(h))

    def h_rotccw_h(self, h):
        return self.h_twin_h(self.h_prev_h(h))

    def h_prev_h(self, h):
        """
        Finds half-edge previous to h by following next cycle.
        Safe for half-edges of non-triangle faces and boundaries.
        """
        h_next = self.h_next_h(h)

        while h_next != h:
            h_prev = h_next
            h_next = self.h_next_h(h_prev)
        return h_prev

    def h_prev_h_by_rot(self, h):
        """
        Finds half-edge previous to h by rotating around origin of h. Faster when length of next cycle is much larger than valence of origin of h (e.g. when h is on a boundary).
        """
        p_h = self.h_twin_h(h)
        n_h = self.h_next_h(p_h)
        while n_h != h:
            p_h = self.h_twin_h(n_h)
            n_h = self.h_next_h(p_h)
        return p_h

    #######################################################
    # Predicates ##########################################
    #######################################################
    def negative_boundary_contains_h(self, h):
        """check if half-edge h is in a negatively oriented boundary of the mesh"""
        return self.f_left_h(h) < 0

    def positive_boundary_contains_h(self, h):
        """check if half-edge h is in a positively oriented boundary of the mesh"""
        return self.f_left_h(self.h_twin_h(h)) < 0

    def boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        # return self.f_left_h(h) < 0 or self.f_left_h(self.h_twin_h(h)) < 0
        return np.logical_or(
            self.negative_boundary_contains_h(h), self.positive_boundary_contains_h(h)
        )

    def boundary_contains_v(self, v):
        """check if vertex v is on the boundary of the mesh"""
        for h in self.generate_H_out_v_clockwise(v):
            if self.f_left_h(h) < 0 or self.f_left_h(self.h_twin_h(h)) < 0:
                return True
        return False

    def h_is_locally_delaunay(self, h):
        r"""
        Checks if edge is locally Delaunay (the circumcircle of the triangle to one side of the edge does not contain the vertex opposite the edge on the triangle's other side)
             vj
             /|\
           vk | vi
             \|/
             vl
        """
        vi = self.v_head_h(self.h_next_h(self.h_twin_h(h)))
        vj = self.v_head_h(h)
        vk = self.v_head_h(self.h_next_h(h))
        vl = self.v_origin_h(h)

        rij = self.xyz_coord_v(vj) - self.xyz_coord_v(vi)
        ril = self.xyz_coord_v(vl) - self.xyz_coord_v(vi)

        rkj = self.xyz_coord_v(vj) - self.xyz_coord_v(vk)
        rkl = self.xyz_coord_v(vl) - self.xyz_coord_v(vk)

        alphai = np.arccos(
            np.dot(rij, ril) / (np.linalg.norm(rij) * np.linalg.norm(ril))
        )
        alphak = np.arccos(
            np.dot(rkl, rkj) / (np.linalg.norm(rkl) * np.linalg.norm(rkj))
        )

        return alphai + alphak <= np.pi

    def h_is_flippable(self, h):
        r"""
        edge flip hlj-->hki is allowed unless hlj is on a boundary or vi and vk are already neighbors
        vj
        /|\
      vk | vi
        \|/
        vl
        """
        if self.boundary_contains_h(h):
            return False
        hlj = h
        hjk = self.h_next_h(hlj)
        # hjl = self.h_twin_h(hlj)
        hli = self.h_next_h(self.h_twin_h(hlj))
        vi = self.v_head_h(hli)
        vk = self.v_head_h(hjk)

        for him in self.generate_H_out_v_clockwise(vi):
            if self.v_head_h(him) == vk:
                return False
        return True

    #######################################################
    # Generators ##########################################
    #######################################################
    def generate_H_out_v_clockwise(self, v, h_start=None):
        """
        Generate outgoing half-edges from vertex v in clockwise order until the starting half-edge is reached again
        """
        if h_start is None:
            h_start = self.h_out_v(v)
        elif self.v_origin_h(h_start) != v:
            raise ValueError("Starting half-edge does not originate at vertex v")
        h = h_start
        while True:
            yield h
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def generate_H_in_v_clockwise(self, v, h_start=None):
        """
        Generate outgoing half-edges from vertex v in clockwise order until the starting half-edge is reached again
        """
        if h_start is None:
            h_start = self.h_in_v(v)
        elif self.v_head_h(h_start) != v:
            raise ValueError("Starting half-edge does not terminate at vertex v")
        h = h_start
        while True:
            yield h
            h = self.h_twin_h(self.h_next_h(h))
            if h == h_start:
                break

    def generate_H_bound_f(self, f):
        """Generate half-edges on the boundary of face f"""
        h = self.h_right_f(f)
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_V_of_f(self, f):
        h = self.h_right_f(f)
        h_start = h
        while True:
            yield self.v_origin_h(h)
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_H_next_h(self, h):
        """Generate half-edges in the face/boundary cycle containing half-edge h"""
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_H_right_b(self, b, h_start=None):
        if h_start is None:
            h_start = self.h_right_b(b)
        elif self.f_left_h(h_start) != -(b + 1):
            raise ValueError(f"half-edge {h_start} is not contained in boundary {b}")
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_F_incident_v_clockwise(self, v, h_start=None):
        """Generate faces incident on vertex v in clockwise order"""
        for h in self.generate_H_out_v_clockwise(v, h_start=h_start):
            if self.negative_boundary_contains_h(h):
                continue
            yield self.f_left_h(h)

    def generate_H_rotcw_h(self, h):
        """
        Generate outgoing half-edges from vertex at the origin of h in clockwise order until the starting half-edge is reached again
        """
        h_start = h
        while True:
            yield h
            h = self.h_rotcw_h(h)
            if h == h_start:
                break

    #######################################################
    # Simplicial Computations ###########################
    #######################################################

    def closure(self, V0, H0, F0, in_place=False):
        """
        Find simplicial closure of (V,H,F) in M by searching F and H. Uses array slicing
        """

        if in_place:
            V, H, F = V0, H0, F0
        else:
            V, H, F = V0.copy(), H0.copy(), F0.copy()
        # next cycle of each face gets
        #   *interior half-edges
        #   *positive boundary half-edges
        arrF = np.array(list(F), dtype=int)
        h_right_F = self.h_right_f(arrF)
        next_h_right_F = self.h_next_h(h_right_F)
        next_next_h_right_F = self.h_next_h(next_h_right_F)
        H.update(h_right_F)
        H.update(next_h_right_F)
        H.update(next_next_h_right_F)
        # twin of interior half-edges gets
        #   *negative boundary half-edges
        #   *any other twins missing from H0
        arrH = np.array(list(H), dtype=int)
        h_twin_H = self.h_twin_h(arrH)
        H.update(h_twin_H)
        # origin of half-edges gets
        #  *vertices missing from V0
        arrH = np.array(list(H), dtype=int)
        v_origin_H = self.v_origin_h(arrH)
        V.update(v_origin_H)
        return V, H, F

    def closure1(self, V, H, F, Hneed2visit=None, Fneed2visit=None, in_place=False):
        """
        Find simplicial closure of (V,H,F) in M by searching F and H.
        """
        # V, H, F = VHF
        if Hneed2visit is None:
            Hneed2visit = H.copy()
        if Fneed2visit is None:
            Fneed2visit = F.copy()
        if in_place:
            closedV, closedH, closedF = V, H, F
        else:
            closedV, closedH, closedF = V.copy(), H.copy(), F.copy()
        # add edges and verts for faces in F
        while Fneed2visit:
            f = Fneed2visit.pop()
            for h in self.generate_H_bound_f(f):
                ht = self.h_twin_h(h)
                v = self.v_origin_h(h)
                closedV.add(v)
                closedH.add(h)
                closedH.add(ht)
                Hneed2visit.discard(h)
                Hneed2visit.discard(ht)
        # add twins and verts for remaining half-edges
        while Hneed2visit:
            h = Hneed2visit.pop()
            ht = self.h_twin_h(h)
            v = self.v_origin_h(h)
            vt = self.v_origin_h(ht)
            closedV.add(v)
            closedV.add(vt)
            closedH.add(ht)
            Hneed2visit.discard(ht)
        return closedV, closedH, closedF

    def star_of_vertex(self, v):
        """Star of a vertex is the set of all simplices that contain the vertex."""
        V = {v}
        H = set()
        F = set()
        for h in self.generate_H_out_v_clockwise(v):
            ht = self.h_twin_h(h)
            H.update([h, ht])
            if not self.negative_boundary_contains_h(h):
                F.add(self.f_left_h(h))

        return V, H, F

    def star_of_edge(self, h):
        """Star of an edge is the set of all simplices that contain the edge."""
        V = set()
        H = {h, self.h_twin_h(h)}
        F = set()
        for hi in H:
            if not self.negative_boundary_contains_h(hi):
                F.add(self.f_left_h(hi))

        return V, H, F

    def star(self, V_in, H_in, F_in):
        """The star of a single simplex is the set of all simplices that have the simplex as a face. The star St(s) of a k-simplex s consists of: s and all (n>k)-simplices that contain s."""
        F = F_in.copy()
        H = H_in.copy()
        V = V_in.copy()

        for h in H_in:
            ht = self.h_twin_h(h)
            H.add(ht)
            if not self.negative_boundary_contains_h(h):
                F.add(self.f_left_h(h))
            if not self.negative_boundary_contains_h(ht):
                F.add(self.f_left_h(ht))
        for v in V_in:
            for h in self.generate_H_out_v_clockwise(v):
                H.add(h)
                H.add(self.h_twin_h(h))
                if not self.negative_boundary_contains_h(h):
                    F.add(self.f_left_h(h))
        return V, H, F

    def link(self, V, H, F):
        """Lk(s)=Cl(St(s))-St(Cl(s))."""
        StCl_V, StCl_H, StCl_F = self.star(*self.closure(V, H, F))
        ClSt_V, ClSt_H, ClSt_F = self.closure(*self.star(V, H, F))
        return ClSt_V - StCl_V, ClSt_H - StCl_H, ClSt_F - StCl_F

    def valence_v(self, v):
        """get the valence of vertex v"""
        valence = 0
        for h in self.generate_H_out_v_clockwise(v):
            valence += 1
        return valence

    def F_incident_b(self, b):
        """get the faces incident on boundary b"""
        F = set()
        for h in self.generate_H_right_b(b):
            v = self.v_origin_h(h)
            F.update(set(self.generate_F_incident_v_clockwise(v, h_start=h)))
        return np.array(list(F), dtype=int)

    def get_unsigned_simplicial_sets(self):
        S0 = {frozenset(v) for v in range(self.num_vertices)}
        S1 = {
            frozenset([self.v_origin_h(h), self.v_head_h(h)])
            for h in range(self.num_half_edges)
        }
        S2 = {frozenset(self.generate_V_of_f(f)) for f in range(self.num_faces)}
        return S0, S1, S2

    #######################################################
    # Mesh modification ##################################
    #######################################################
    # def rigid_transform(self, translation, angle_vec, origin=None):
    #     """
    #     Apply a rigid transformation to the mesh.
    #     t = translation in R3
    #     w = angle_vec in R3~so3
    #     R = exp_so3(w) in SO3
    #     o = origin in R3~E3
    #     x->o+R*(x-o)+t, or x->R*x+t if o is not provided
    #     """
    #     self.xyz_coord_V = rigid_transform(
    #         translation, angle_vec, self.xyz_coord_V, origin=origin
    #     )

    def update_vertex(self, v, xyz=None, h_out=None):
        if xyz is not None:
            self.xyz_coord_V[v] = xyz
        if h_out is not None:
            self.h_out_V_[v] = h_out

    def update_half_edge(self, h, h_next=None, h_twin=None, v_origin=None, f_left=None):
        if h_next is not None:
            self.h_next_H_[h] = h_next
        if h_twin is not None:
            self.h_twin_H_[h] = h_twin
        if v_origin is not None:
            self.v_origin_H_[h] = v_origin
        if f_left is not None:
            self.f_left_H_[h] = f_left

    def update_face(self, f, h_bound=None):
        if h_bound is not None:
            self.h_right_F_[f] = h_bound

    def flip_edge(self, h):
        r"""
        h cannot be on boundary!
                v1                           v1
              /    \                       /  |  \
             /      \                     /   |   \
            /h3    h2\                   /h3  |  h2\
           /    f0    \                 /     |     \
          /            \               /  f0  |  f1  \
         /      h0      \             /       |       \
        v2--------------v0  |----->  v2     h0|h1     v0
         \      h1      /             \       |       /
          \            /               \      |      /
           \    f1    /                 \     |     /
            \h4    h5/                   \h4  |  h5/
             \      /                     \   |   /
              \    /                       \  |  /
                v3                           v3
        v0
        --
        h_out
            pre-flip: may be h1
            post-flip: set to h2 if needed
        v2
        --
            pre-flip: may be h0
            post-flip: set to h4 if needed
        h0
        --
        v_origin_h(h0)
            pre-flip: v2
            post-flip: v3
        h_next
            pre-flip: h2
            post-flip: h3
        h_twin
            unchanged
        f_left
            unchanged
        h1
        --
        v_origin
            pre-flip: v0
            post-flip: v1
        h_next
            pre-flip: h4
            post-flip: h5
        h_twin
            unchanged
        f_left
            unchanged
        h2
        --
        v_origin
            unchanged
        h_next
            pre-flip: h3
            post-flip: h1
        h_twin
            unchanged
        f_left
            pre-flip: f0
            post-flip: f1
        h3
        --
        v_origin
            unchanged
        h_next
            pre-flip: h0
            post-flip: h4
        h_twin
            unchanged
        f_left
            unchanged
        h4
        --
        v_origin
            unchanged
        h_next
            pre-flip: h5
            post-flip: h0
        h_twin
            unchanged
        f_left
            pre-flip: f1
            post-flip: f0
        h5
        --
        v_origin
            unchanged
        h_next
            pre-flip: h1
            post-flip: h2
        h_twin
            unchanged
        f_left
            unchanged
        f0
        --
        h_bound
            pre-flip: may be h2
            post-flip: set to h3 if needed
        f1
        --
        h_bound
            pre-flip: may be h4
            post-flip: set to h5 if needed
        """
        # get involved half-edges/vertices/faces
        h0 = h
        h1 = self.h_twin_h(h0)
        h2 = self.h_next_h(h0)
        h3 = self.h_next_h(h2)
        h4 = self.h_next_h(h1)
        h5 = self.h_next_h(h4)

        v0 = self.v_origin_h(h1)
        v1 = self.v_origin_h(h3)
        v2 = self.v_origin_h(h0)
        v3 = self.v_origin_h(h5)

        f0 = self.f_left_h(h0)
        f1 = self.f_left_h(h1)

        # update vertices
        if self.h_out_v(v0) == h1:
            self.update_vertex(v0, h_out=h2)
        if self.h_out_v(v2) == h0:
            self.update_vertex(v2, h_out=h4)
        # update half-edges
        self.update_half_edge(h0, v_origin=v3, h_next=h3)
        self.update_half_edge(h1, v_origin=v1, h_next=h5)
        self.update_half_edge(h2, h_next=h1, f_left=f1)
        self.update_half_edge(h3, h_next=h4)
        self.update_half_edge(h4, h_next=h0, f_left=f0)
        self.update_half_edge(h5, h_next=h2)
        # update faces
        if self.h_right_f(f0) == h2:
            self.update_face(f0, h_bound=h3)
        if self.h_right_f(f1) == h4:
            self.update_face(f1, h_bound=h5)

    def flip_non_delaunay(self):
        flip_count = 0
        for h in range(self.num_half_edges):
            if not self.h_is_locally_delaunay(h):
                if self.h_is_flippable(h):
                    self.flip_edge(h)
                    flip_count += 1
        return flip_count

    def smooth_graph_laplacian(self, weight=0.25, smooth_boundary=False):
        """ """
        Q = self.xyz_coord_V
        if smooth_boundary:
            lapQ = self.graph_laplacian(Q)

        else:
            lapQ = np.zeros_like(Q)
            for i in range(self.num_vertices):
                if self.boundary_contains_v(i):
                    # lapQ[i] = Q[i]
                    continue
                deg = 0
                for h in self.generate_H_out_v_clockwise(i):
                    lapQ[i] += Q[self.v_head_h(h)]
                    deg += 1

                lapQ[i] /= deg
                lapQ[i] -= Q[i]

        self.xyz_coord_V += weight * lapQ

    def taubin_smooth_graph_laplacian(
        self, weight_shrink=0.25, weight_inflate=-0.25, smooth_boundary=False
    ):
        self.smooth_graph_laplacian(
            weight=weight_shrink, smooth_boundary=smooth_boundary
        )
        self.smooth_graph_laplacian(
            weight=weight_inflate, smooth_boundary=smooth_boundary
        )

    def update_V_slice(self, index_slice, xyz_coord_V=None, h_out_V=None):
        """ """
        if xyz_coord_V is not None:
            self.xyz_coord_V[index_slice] = xyz_coord_V
        if h_out_V is not None:
            self.h_out_V[index_slice] = h_out_V

    def update_H_slice(
        self, index_slice, h_next_H=None, h_twin_H=None, v_origin_H=None, f_left_H=None
    ):
        """ """
        if h_next_H is not None:
            self.h_next_H[index_slice] = h_next_H
        if h_twin_H is not None:
            self.h_twin_H[index_slice] = h_twin_H
        if v_origin_H is not None:
            self.v_origin_H[index_slice] = v_origin_H
        if f_left_H is not None:
            self.f_left_H[index_slice] = f_left_H

    def update_F_slice(self, index_slice, h_right_F=None):
        """ """
        if h_right_F is not None:
            self.h_right_F[index_slice] = h_right_F

    def divide_face_barycentric(self, f):
        dNv = 1
        dNh = 6
        dNf = 2
        # dNb = 0
        self.xyz_coord_V = np.concatenate([self.xyz_coord_V, np.zeros((dNv, 3))])
        self.h_out_V = np.concatenate([self.h_out_V, np.zeros(dNv, dtype=int)])
        self.v_origin_H = np.concatenate([self.v_origin_H, np.zeros(dNh, dtype=int)])
        self.h_next_H = np.concatenate([self.h_next_H, np.zeros(dNh, dtype=int)])
        self.h_twin_H = np.concatenate([self.h_twin_H, np.zeros(dNh, dtype=int)])
        self.f_left_H = np.concatenate([self.f_left_H, np.zeros(dNh, dtype=int)])
        self.h_right_F = np.concatenate([self.h_right_F, np.zeros(dNf, dtype=int)])
        # self.h_negative_B = np.concatenate([self.h_negative_B, np.zeros(dNb, dtype=int)])

        # Get/create exsisting/new vertices, half-edges, faces, boundaries involved in the operation
        f0 = f
        h0 = self.h_right_f(f0)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        v0 = self.v_origin_h(h0)
        v1 = self.v_origin_h(h1)
        v2 = self.v_origin_h(h2)
        V = np.concatenate(
            [[v0, v1, v2], list(range(self.num_vertices, self.num_vertices + dNv))],
            dtype=int,
        )
        H = np.concatenate(
            [[h0, h1, h2], list(range(self.num_half_edges, self.num_half_edges + dNh))],
            dtype=int,
        )
        F = np.concatenate(
            [[f0], list(range(self.num_faces, self.num_faces + dNf))], dtype=int
        )

        #####
        self.update_vertex(
            V[3], xyz=np.sum(self.xyz_coord_V[V[:3]], axis=0), h_out=H[4]
        )
        self.update_H_slice(H[3:], v_origin_H=[V[1], V[3], V[2], V[3], V[0], V[3]])
        self.update_H_slice(
            H, h_next_H=[H[3], H[5], H[7], H[8], H[1], H[4], H[2], H[6], H[0]]
        )
        self.update_H_slice(
            H[3:], h_next_H=None, h_twin_H=[H[4], H[3], H[6], H[5], H[8], H[7]]
        )
        self.update_H_slice(
            H[1:], f_left_H=[F[1], F[2], F[0], F[1], F[1], F[2], F[2], F[0]]
        )
        self.update_F_slice(F, h_right_F=H[:3])
        return dNv, dNh, dNf

    def uniform_flip_sweep(self, r0=0.1):
        num_half_edges = self.get_num_half_edges()

        H = np.random.permutation(num_half_edges)
        for h in H:
            r = np.random.rand()
            if r > r0:
                continue
            if self.h_is_flippable(h):
                self.flip_edge(h)

    #######################################################
    # Geometric Computations #############################
    #######################################################
    def vec_area_f(self, f):
        h0 = self.h_right_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        x0 = self.xyz_coord_v(self.v_origin_h(h0))
        x1 = self.xyz_coord_v(self.v_origin_h(h1))
        x2 = self.xyz_coord_v(self.v_origin_h(h2))
        vec_area = 0.5 * (np.cross(x0, x1) + np.cross(x1, x2) + np.cross(x2, x0))
        return vec_area

    def area_f(self, f):
        return np.linalg.norm(self.vec_area_f(f))

    def area_F(self):
        N = self.num_faces
        A = np.zeros(N, dtype=float)
        for k in range(N):
            A[k] = self.area_f(k)
        return A

    def length_h(self, h):
        v0 = self.v_origin_h(h)
        v1 = self.v_head_h(h)
        return np.linalg.norm(self.xyz_coord_v(v1) - self.xyz_coord_v(v0))

    def length_H(self):
        L = np.zeros(self.num_half_edges, dtype=float)
        for h in range(self.num_half_edges):
            L[h] = self.length_h(h)
        return L

    def total_area_of_faces(self):
        Atot = 0.0
        for f in range(self.num_faces):
            Atot += self.area_f(f)

        return Atot

    def barcell_area(self, v):
        """area of the barycentric cell dual to vertex v"""
        # r = self.xyz_coord_v(v)
        A = 0.0
        for h in self.generate_H_out_v_clockwise(v):
            if self.negative_boundary_contains_h(h):
                continue
            # r1 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(h)))
            # r2 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(self.h_next_h(h))))
            # A_face_vec = (
            #     np.cross(r, r1) / 2 + np.cross(r1, r2) / 2 + np.cross(r2, r) / 2
            # )
            # A_face = np.sqrt(
            #     A_face_vec[0] ** 2 + A_face_vec[1] ** 2 + A_face_vec[2] ** 2
            # )

            A += self.area_f(self.f_left_h(h)) / 3

        return A

    def total_area_of_dual_barcells(self):
        Atot = 0.0
        for v in range(self.num_vertices):
            Atot += self.barcell_area(v)

        return Atot

    def vorcell_area(self, v):
        r"""area of voronoi cell dual to vertex v
                  v=v0
                //  \\
               // |  \\
              // /|   \\
             //   ||   \\
            //    || h20\\
           //     ||     \\
        ...       ||h01    v2
           \\     ||     //
            \\    || h12//
             \\   ||   //
              \\  ||/ //
               \\ |  //
                \\| //
                  v1
        """
        Atot = 0.0
        r0 = self.xyz_coord_v(v)
        for h in self.generate_H_out_v_clockwise(v):
            if self.negative_boundary_contains_h(h):
                continue
            r1 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(h)))
            r2 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(self.h_next_h(h))))
            r01 = r1 - r0
            r12 = r2 - r1
            r20 = r0 - r2

            norm_r20 = np.linalg.norm(r20)
            norm_r01 = np.linalg.norm(r01)
            norm_r12 = np.linalg.norm(r12)
            cos_210 = -np.dot(r12, r01) / (norm_r12 * norm_r01)
            cos_021 = -np.dot(r20, r12) / (norm_r20 * norm_r12)

            cot_210 = cos_210 / np.sqrt(1 - cos_210**2)
            cot_021 = cos_021 / np.sqrt(1 - cos_021**2)
            Atot += norm_r20**2 * cot_210 / 8 + norm_r01**2 * cot_021 / 8

        return Atot

    def vorcell_area_V(self):

        N = self.num_vertices
        A = np.zeros(N, dtype=float)
        for k in range(N):
            A[k] = self.vorcell_area(k)
        return A

    def total_area_of_dual_vorcells(self):
        Atot = 0.0
        for v in range(self.num_vertices):
            Atot += self.vorcell_area(v)

        return Atot

    def meyercell_area(self, v):
        """Meyer's mixed area of cell dual to vertex v"""
        Atot = 0.0
        ri = self.xyz_coord_v(v)
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        # h_start = self.V_hedge[v]
        # hij = h_start
        # while True:
        for hij in self.generate_H_out_v_clockwise(v):
            if self.negative_boundary_contains_h(hij):
                continue
            hjjp1 = self.h_next_h(hij)
            hjp1i = self.h_next_h(hjjp1)
            vj = self.v_origin_h(hjjp1)
            rj = self.xyz_coord_v(vj)
            vjp1 = self.v_origin_h(hjp1i)
            rjp1 = self.xyz_coord_v(vjp1)

            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
            rjp1_ri = rjp1[0] * ri[0] + rjp1[1] * ri[1] + rjp1[2] * ri[2]

            normDrij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
            # normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
            normDrjjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
            normDrjp1i = np.sqrt(rjp1_rjp1 - 2 * rjp1_ri + ri_ri)
            cos_thetajijp1 = (ri_ri + rj_rjp1 - ri_rj - rjp1_ri) / (
                normDrij * normDrjp1i
            )
            cos_thetajp1ji = (rj_rj + rjp1_ri - rj_rjp1 - ri_rj) / (
                normDrij * normDrjjp1
            )
            cos_thetaijp1j = (rjp1_rjp1 + ri_rj - rj_rjp1 - rjp1_ri) / (
                normDrjp1i * normDrjjp1
            )
            if cos_thetajijp1 < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += (
                    np.sqrt(
                        semiP
                        * (semiP - normDrij)
                        * (semiP - normDrjjp1)
                        * (semiP - normDrjp1i)
                    )
                    / 2
                )
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 4
            elif cos_thetajp1ji < 0 or cos_thetaijp1j < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += (
                    np.sqrt(
                        semiP
                        * (semiP - normDrij)
                        * (semiP - normDrjjp1)
                        * (semiP - normDrjp1i)
                    )
                    / 4
                )
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 8
            else:
                cot_thetaijp1j = cos_thetaijp1j / np.sqrt(1 - cos_thetaijp1j**2)
                cot_thetajp1ji = cos_thetajp1ji / np.sqrt(1 - cos_thetajp1ji**2)
                Atot += (
                    normDrij**2 * cot_thetaijp1j / 8
                    + normDrjp1i**2 * cot_thetajp1ji / 8
                )

        return Atot

    def total_area_of_dual_meyercells(self):
        Atot = 0.0
        for v in range(self.num_vertices):
            Atot += self.meyercell_area(v)

        return Atot

    def barcell_area_V(self):

        N = self.num_vertices
        A = np.zeros(N, dtype=float)
        for k in range(N):
            A[k] = self.barcell_area(k)
        return A

    def total_volume(self):
        Nf = self.num_faces
        vol = 0.0
        for f in range(Nf):
            h0 = self.h_right_f(f)
            h1 = self.h_next_h(h0)
            h2 = self.h_next_h(h1)
            v0 = self.v_origin_h(h0)
            v1 = self.v_origin_h(h1)
            v2 = self.v_origin_h(h2)
            x0 = self.xyz_coord_v(v0)
            x1 = self.xyz_coord_v(v1)
            x2 = self.xyz_coord_v(v2)
            vol_f = np.dot(x0, np.cross(x1, x2)) / 6
            vol += vol_f
        return abs(vol)

    def average_edge_length(self):
        return np.mean(self.length_H())

    def average_face_area(self):
        return self.total_area_of_faces() / self.num_faces

    ##############
    # unit normals
    def normal_v(self, i):
        """default vertex unit normal"""
        return self.normal_other_weighted_v(i)

    def normal_some_face_of_v(self, i):
        h = self.h_out_v(i)
        f = self.f_left_h(h)
        if f < 0:
            h = self.h_rotcw_h(h)
            f = self.f_left_h(h)
        avec = self.vec_area_f(f)
        n = avec / np.linalg.norm(avec)
        return n

    def normal_some_face_of_V(self):
        n = np.zeros((self.num_vertices, 3), dtype=float)
        for i in range(self.num_vertices):
            n[i] = self.normal_some_face_of_v(i)
        return n

    def normal_other_weighted_v(self, i):
        """Weights for Computing Vertex Normals from Facet Normals Max99"""
        n = np.zeros(3)
        x = self.xyz_coord_v(i)
        h = self.h_out_v(i)
        rrot = self.xyz_coord_v(self.v_head_h(h)) - x
        h = self.h_rotcw_h(h)
        for hrot in self.generate_H_out_v_clockwise(i, h_start=h):
            r = rrot
            jrot = self.v_head_h(hrot)
            rrot = self.xyz_coord_v(jrot) - x
            if self.negative_boundary_contains_h(hrot):
                continue
            n += np.cross(rrot, r) / (np.dot(r, r) * np.dot(rrot, rrot))
        n /= np.linalg.norm(n)
        return n

    def normal_other_weighted_V(self):
        n = np.zeros((self.num_vertices, 3), dtype=float)
        for i in range(self.num_vertices):
            n[i] = self.normal_other_weighted_v(i)
        return n

    def normal_laplacian_V(self):
        """
        Compute unit normals from mean curvature vector at all vertices
        """
        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)

        return n

    #######################################################
    # Differential operators ##############################
    #######################################################
    def laplacian(self, Q):
        """
        overwrite to set which laplacian to use
        """
        return self.cotan_laplacian(Q)

    def cotan_laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex
        """
        # Nv = self.num_vertices
        lapQ = np.zeros_like(Q)
        for vi in range(self.num_vertices):
            Atot = 0.0
            ri = self.xyz_coord_v(vi)
            qi = Q[vi]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            for hij in self.generate_H_out_v_clockwise(vi):
                hijm1 = self.h_next_h(self.h_twin_h(hij))
                hijp1 = self.h_twin_h(self.h_prev_h(hij))
                vjm1 = self.v_head_h(hijm1)
                vj = self.v_head_h(hij)
                vjp1 = self.v_head_h(hijp1)

                qj = Q[vj]

                rjm1 = self.xyz_coord_v(vjm1)
                rj = self.xyz_coord_v(vj)
                rjp1 = self.xyz_coord_v(vjp1)

                rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
                rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
                ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
                rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

                Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
                Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
                Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
                Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
                Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

                cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

                cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

                cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
                cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

                Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
                lapQ[vi] += (cot_thetam + cot_thetap) * (qj - qi) / 2
            lapQ[vi] /= Atot

        return lapQ

    def graph_laplacian(self, Q):
        """
        Computes the graph Laplacian of Q at each vertex
        """
        lapQ = np.zeros_like(Q)
        for i in range(self.num_vertices):
            deg = 0
            for h in self.generate_H_out_v_clockwise(i):
                lapQ[i] += Q[self.v_head_h(h)]
                deg += 1

            lapQ[i] /= deg
            lapQ[i] -= Q[i]

        return lapQ

    #######################################################
    # Miscellaneous ######################################
    #######################################################
    def vf_samples(self):

        return (self.xyz_coord_V, self.V_cycle_F)
