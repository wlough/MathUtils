# from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
# from src_python.pybrane.half_edge_patch import HalfEdgePatch
# from src_python.pybrane_graphics.mesh_viewer import MeshViewer
# import numpy as np
#


# %%
##################################################
# half edge hexagon
##################################################
def half_edge_hexagon():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer

    ply_path = "data/convergence_tests/nice_plane/plane_0000006_he.ply"
    viewer_kwargs = {
        "image_dir": "output/half_edge_hexagon",
        "image_index_length": 6,
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 1.0),
        "rgb_background": (1, 1, 1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 4.0,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    m.xyz_coord_V
    mv = MeshViewer(m, **viewer_kwargs)
    mv.plot(save=True, show=False)


half_edge_hexagon()


# %%
##################################################
# next cycle
##################################################
def next_cycle():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer

    ply_path = "data/convergence_tests/nice_plane/plane_0000006_he.ply"
    viewer_kwargs = {
        "image_index_length": 6,
        "image_dir": "output/next_cycle",
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 0.1),
        "rgb_background": (1, 1, 1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 4.0,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    m.xyz_coord_V
    rgba = (0.5804, 0.0, 0.8275, 1.0)
    rgba_clear = (0.5804, 0.0, 0.8275, 0.1)
    mv = MeshViewer(m, **viewer_kwargs)
    # h0 = 0
    # h1 = m.h_next_h(h0)
    # h2 = m.h_next_h(h1)
    # mv = MeshViewer(m, **viewer_kwargs)
    # mv.update_rgba_H(rgba, [h0])
    # mv.plot()
    # mv.update_rgba_H(rgba_clear, [h0])
    # mv.update_rgba_H(rgba, [h1])
    # mv.plot()
    # mv.update_rgba_H(rgba_clear, [h1])
    # mv.update_rgba_H(rgba, [h2])
    # mv.plot()
    h = 0
    for _ in range(5):
        mv.update_rgba_H(rgba, [h])
        mv.plot(save=True, show=False)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_next_h(h)


next_cycle()


# %%
##################################################
# twin cycle
##################################################
def twin_cycle():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer

    ply_path = "data/convergence_tests/nice_plane/plane_0000006_he.ply"
    viewer_kwargs = {
        "image_index_length": 6,
        "image_dir": "output/twin_cycle",
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 0.1),
        "rgb_background": (1, 1, 1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 4.0,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    m.xyz_coord_V
    rgba = (0.5804, 0.0, 0.8275, 1.0)
    rgba_clear = (0.5804, 0.0, 0.8275, 0.1)
    h0 = 0
    h1 = m.h_twin_h(h0)
    mv = MeshViewer(m, **viewer_kwargs)
    # mv.update_rgba_H(rgba, [h0])
    # mv.plot()
    # mv.update_rgba_H(rgba_clear, [h0])
    # mv.update_rgba_H(rgba, [h1])
    # mv.plot()
    h = 0
    for _ in range(12):
        mv.update_rgba_H(rgba, [h])
        mv.plot(save=True, show=False)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_twin_h(h)


twin_cycle()


# %%
##################################################
# rot cycle
##################################################
def rot_cycle():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer

    ply_path = "data/convergence_tests/nice_plane/plane_0000006_he.ply"
    viewer_kwargs = {
        "image_index_length": 6,
        "image_dir": "output/rot_cycle",
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 0.1),
        "rgb_background": (1, 1, 1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 4.0,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    mv = MeshViewer(m, **viewer_kwargs)
    rgba = (0.5804, 0.0, 0.8275, 1.0)
    rgba_clear = (0.5804, 0.0, 0.8275, 0.1)
    # h0 = 0
    # h1 = m.h_next_h(m.h_twin_h(h0))
    # h2 = m.h_next_h(m.h_twin_h(h1))
    # h3 = m.h_next_h(m.h_twin_h(h2))
    # h4 = m.h_next_h(m.h_twin_h(h3))
    # h5 = m.h_next_h(m.h_twin_h(h4))
    # H = [h0, h1, h2, h3, h4, h5]
    #
    # for h in H:
    #     mv.update_rgba_H(rgba, [h])
    #     mv.plot()
    #     mv.update_rgba_H(rgba_clear, H)
    h = 0
    for _ in range(13):
        mv.update_rgba_H(rgba, [h])
        mv.plot(save=True, show=False)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)


rot_cycle()


# %%
##################################################
# Front propagation
##################################################
def front_propagation_nice():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh

    # from src_python.pybrane.half_edge_patch import HalfEdgePatch
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer
    import numpy as np
    from src_python.pybrane_graphics.pretty_pictures import get_cmap

    # fig_count = 0
    R = np.linspace(0.4, 2, 4)
    cmap = get_cmap(R[0], R[-1], "coolwarm_r")
    colors = np.array([cmap(_) for _ in R])
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    color3 = colors[3]

    ply_path = "data/convergence_tests/nice_plane/plane_0000096_he.ply"
    ply_path = "data/convergence_tests/nice_plane/plane_0000384_he.ply"
    viewer_kwargs = {
        "rgb_background": (1, 1, 1),
        "image_index_length": 6,
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 0.1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 2.25,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    mv = MeshViewer(m, **viewer_kwargs)
    rgba = (0.5804, 0.0, 0.8275, 1.0)
    rgba_clear = (0.5804, 0.0, 0.8275, 0.1)
    # color0 = (0.9, 0.15, 0.15, 0.8)
    # color1 = (0.45, 0.15, 0.15, 0.8)
    # color2 = (0.0, 0.15, 0.9, 0.8)
    # ring 0 ##################################
    F0 = set()
    F = set()
    h_start = 0
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f in F:
            break
        F.add(f)
        mv.update_rgba_F(color0, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)

    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)
    F0.update(F)
    #
    ###########################################
    # ring 1 ##################################
    ###########################################
    Fold = F.copy()
    F = set()
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f in Fold:
            h = m.h_rotccw_h(h)
            h = m.h_next_h(h)
            h = m.h_rotcw_h(h)
            f = m.f_left_h(h)
        if f in F:
            break
        F.add(f)
        mv.update_rgba_F(color1, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)

    h_start = m.h_rotccw_h(h)
    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)
    F0.update(F)
    ###########################################
    # ring 2 ##################################
    ###########################################
    Fold = F.copy()
    F = set()
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f in Fold:
            h = m.h_rotccw_h(h)
            h = m.h_next_h(h)
            h = m.h_rotcw_h(h)
            f = m.f_left_h(h)
        if f in F:
            break
        F.add(f)
        mv.update_rgba_F(color2, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)

    h_start = m.h_rotccw_h(h)
    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)
    F0.update(F)
    ###########################################
    # ring 3 ##################################
    ###########################################

    Fold = F.copy()
    F = set()
    h = h_start
    # first_step = True
    while True:
        f = m.f_left_h(h)
        # if f in Fold:
        while f in Fold:
            h = m.h_rotccw_h(h)
            h = m.h_next_h(h)
            h = m.h_rotcw_h(h)
            f = m.f_left_h(h)
        if f in F:
            break
        # if not first_step:
        #     if h == h_start:
        #         break
        F.add(f)
        mv.update_rgba_F(color3, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)
        # first_step = False

    h_start = m.h_rotccw_h(h)
    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)

    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)


def front_propagation_nonhomogeneous():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh

    # from src_python.pybrane.half_edge_patch import HalfEdgePatch
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer

    import numpy as np
    from src_python.pybrane_graphics.pretty_pictures import get_cmap

    # fig_count = 0
    R = np.linspace(0.4, 2, 4)
    cmap = get_cmap(R[0], R[-1], "coolwarm_r")
    colors = np.array([cmap(_) for _ in R])
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    color3 = colors[3]

    ply_path = "data/convergence_tests/nonhomogeneous_plane/plane_0000772_he.ply"
    viewer_kwargs = {
        "rgb_background": (1, 1, 1),
        "image_index_length": 6,
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 0.1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 2.25,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    mv = MeshViewer(m, **viewer_kwargs)
    rgba = (0.5804, 0.0, 0.8275, 1.0)
    rgba_clear = (0.5804, 0.0, 0.8275, 0.1)
    # color0 = (0.9, 0.15, 0.15, 0.8)
    # color1 = (0.45, 0.15, 0.15, 0.8)
    # color2 = (0.0, 0.15, 0.9, 0.8)
    # ring 0 ##################################
    F0 = set()
    F = set()
    h_start = 0
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f in F:
            break
        F.add(f)
        mv.update_rgba_F(color0, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)

    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)
    F0.update(F)
    #
    ###########################################
    # ring 1 ##################################
    ###########################################
    Fold = F.copy()
    F = set()
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f in Fold:
            h = m.h_rotccw_h(h)
            h = m.h_next_h(h)
            h = m.h_rotcw_h(h)
            f = m.f_left_h(h)
        if f in F:
            break
        F.add(f)
        mv.update_rgba_F(color1, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)

    h_start = m.h_rotccw_h(h)
    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)
    F0.update(F)
    ###########################################
    # ring 2 ##################################
    ###########################################
    Fold = F.copy()
    F = set()
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f in Fold:
            h = m.h_rotccw_h(h)
            h = m.h_next_h(h)
            h = m.h_rotcw_h(h)
            f = m.f_left_h(h)
        if f in F:
            break
        F.add(f)
        mv.update_rgba_F(color2, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)

    h_start = m.h_rotccw_h(h)
    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)
    F0.update(F)
    ###########################################
    # ring 3 ##################################
    ###########################################
    Hbdry = set()
    for f in F0:
        h0 = m.h_right_f(f)
        h1 = m.h_next_h(h0)
        h2 = m.h_next_h(h1)
        for h in [h0, h1, h2]:
            ht = m.h_twin_h(h)
            ft = m.f_left_h(ht)
            if ft not in F0:
                Hbdry.add(ht)

    # Hbdry.discard(1771)
    # mv.update_rgba_H(color0, [606])
    # mv.update_rgba_H(color1, [1681])
    # mv.update_rgba_H(color2, [1771])
    # mv.plot(show=True, save=False)
    # print(h_start)
    Fold = F.copy()
    F = set()
    h = h_start
    first_step = True
    while Hbdry:
        print(Hbdry)
        f = m.f_left_h(h)
        # if f in Fold:
        while f in Fold:
            h = m.h_rotccw_h(h)
            Hbdry.discard(h)
            h = m.h_next_h(h)
            Hbdry.discard(h)
            h = m.h_rotcw_h(h)
            Hbdry.discard(h)
            f = m.f_left_h(h)
        # if f in F:
        #     continue

        if not first_step:
            if h == h_start:
                break
        Hbdry.discard(h)
        F.add(f)
        mv.update_rgba_F(color3, [f])
        mv.update_rgba_H(rgba, [h])
        mv.plot(show=False, save=True)
        mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)
        first_step = False

    h_start = m.h_rotccw_h(h)
    h_start = m.h_next_h(h_start)
    h_start = m.h_rotcw_h(h_start)

    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)


def front_propagation_nonhomogeneous2():
    from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
    from src_python.pybrane_graphics.mesh_viewer import MeshViewer
    import numpy as np
    from src_python.pybrane_graphics.pretty_pictures import get_cmap

    R = np.linspace(0.4, 2, 4)
    cmap = get_cmap(R[0], R[-1], "coolwarm_r")
    colors = np.array([cmap(_) for _ in R])
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    color3 = colors[3]

    ply_path = "data/convergence_tests/nonhomogeneous_plane/plane_0000772_he.ply"
    viewer_kwargs = {
        "rgb_background": (1, 1, 1),
        "image_index_length": 6,
        "show_wireframe_surface": True,
        "show_face_colored_surface": True,
        "show_vertices": False,
        "show_half_edges": True,
        "rgba_face": (1.0, 0.7431, 0.0, 1.0),
        "rgba_edge": (0.0, 0.0, 0.0, 1.0),
        "rgba_half_edge": (0.5804, 0.0, 0.8275, 0.1),
        "view": {
            "azimuth": 0.0,
            "elevation": 0.0,
            "distance": 2.25,
            "focalpoint": (0, 0, 0),
        },
    }
    m = HalfEdgeMesh.from_he_ply(ply_path)
    mv = MeshViewer(m, **viewer_kwargs)
    rgba = (0.5804, 0.0, 0.8275, 1.0)
    rgba_clear = (0.5804, 0.0, 0.8275, 0.1)
    # ring 0 ###############################
    F_all = set()
    F_frontier = set()
    v_start = m.v_origin_h(0)
    h_start = m.h_out_v(v_start)
    h = h_start
    while True:
        f = m.f_left_h(h)
        if f not in F_all:
            F_frontier.add(f)
            mv.update_rgba_F(color0, [f])
            mv.update_rgba_H(rgba, [h])
            mv.plot(show=False, save=True)
            mv.update_rgba_H(rgba_clear, [h])
        h = m.h_rotcw_h(h)
        if h == h_start:
            break

    F_all.update(F_frontier)
    H_bdry = set()
    for f in F_frontier:
        h0 = m.h_right_f(f)
        h1 = m.h_next_h(h0)
        h2 = m.h_next_h(h1)
        for h in (h0, h1, h2):
            ht = m.h_twin_h(h)
            ft = m.f_left_h(ht)
            if ft not in F_all:
                H_bdry.add(ht)

    ##################################
    for color in [color1, color2, color3]:
        F_frontier = set()
        while H_bdry:
            h_start = H_bdry.pop()
            h = h_start
            while True:
                f = m.f_left_h(h)
                if f not in F_all:
                    F_frontier.add(f)
                    F_all.add(f)
                    mv.update_rgba_F(color, [f])
                    mv.update_rgba_H(rgba, [h])
                    mv.plot(show=False, save=True)
                    mv.update_rgba_H(rgba_clear, [h])
                h = m.h_rotcw_h(h)
                if h == h_start:
                    break

        F_all.update(F_frontier)
        H_bdry = set()
        for f in F_frontier:
            h0 = m.h_right_f(f)
            h1 = m.h_next_h(h0)
            h2 = m.h_next_h(h1)
            for h in (h0, h1, h2):
                ht = m.h_twin_h(h)
                ft = m.f_left_h(ht)
                if ft not in F_all:
                    H_bdry.add(ht)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)
    mv.plot(show=False, save=True)


front_propagation_nice()
# front_propagation_nonhomogeneous2()


# %%
# %%
##################################################
# vertex dual area
##################################################
# def dual_area():
from src_python.pybrane.half_edge_mesh import HalfEdgeMesh
from src_python.pybrane_graphics.mesh_viewer import MeshViewer

ply_path = "data/convergence_tests/nice_plane/plane_0000006_he.ply"
viewer_kwargs = {
    "image_dir": "output/vertex_dual_area",
    "image_index_length": 6,
    "show_wireframe_surface": True,
    "show_face_colored_surface": True,
    "show_vertices": True,
    "show_half_edges": False,
    # "show_face_colored_surface": False,
    # "show_vertex_colored_surface": True,
    "radius_vertex": 0.125,
    "rgba_vertex": (0.7057, 0.0156, 0.1502, 1.0),
    "rgba_face": (1.0, 0.7431, 0.0, 1.0),
    "rgba_edge": (0.0, 0.0, 0.0, 1.0),
    "rgba_half_edge": (0.5804, 0.0, 0.8275, 1.0),
    "rgb_background": (1, 1, 1),
    "view": {
        "azimuth": 0.0,
        "elevation": 0.0,
        "distance": 4.0,
        "focalpoint": (0, 0, 0),
    },
}
m = HalfEdgeMesh.from_he_ply(ply_path)
m.xyz_coord_V
mv = MeshViewer(m, **viewer_kwargs)
# mv.update_rgba_V((0.7057, 0.0156, 0.1502, 1.0), [0])
mv.update_rgba_V((0.0, 0.4471, 0.6980, 1.0), [0])
mv.plot(save=True, show=False)
