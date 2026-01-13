import numpy as np
import warnings
import pickle
import gzip
import os
import subprocess


def make_output_dir(output_dir, overwrite=False, sub_dirs=None):
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError(
            f"{output_dir} already exists. Choose a different output_dir, or set overwrite=True"
        )
    os.system(f"mkdir -p {output_dir}")
    if hasattr(sub_dirs, "__iter__"):
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(output_dir, sub_dir)
            os.system(f"mkdir -p {sub_dir_path}")


def chunk_file_with_split(filename, chunk_size="40M"):
    try:
        subprocess.run(
            ["split", "-b", chunk_size, filename, f"{filename}.part"], check=True
        )
        print(f"File {filename} has been chunked into {chunk_size} pieces.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while chunking the file: {e}")


def unchunk_file_with_cat(filename, output_filename):
    try:
        command = f"cat {filename}.part* > {output_filename}"
        subprocess.run(command, shell=True, check=True)
        print(f"Chunked files have been recombined into {output_filename}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while unchunking the files: {e}")


def save_pkl(data, filename, compressed=False, chunk=False, remove_unchunked=False):
    if compressed:
        with gzip.open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(filename)
    max_size = 40 * 1024 * 1024  # 50MB in bytes

    if file_size > max_size:
        warnings.warn(
            f"The file {filename} exceeds 40MB. Its size is {file_size / (1024 * 1024):.2f}MB.",
            UserWarning,
        )
        if chunk:
            chunk_file_with_split(filename, chunk_size="40M")
            if remove_unchunked:
                os.remove(filename)


def load_pkl(filename, compressed=False):
    if compressed:
        with gzip.open(filename, "rb") as f:
            data = pickle.load(f)
    else:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    return data


def save_npz(data, filename, compressed=False, chunk=False, remove_unchunked=False):
    if compressed:
        np.savez_compressed(filename, **data)
    else:
        np.savez(filename, **data)

    file_size = os.path.getsize(filename)
    max_size = 40 * 1024 * 1024  # 50MB in bytes

    if file_size > max_size:
        warnings.warn(
            f"The file {filename} exceeds 40MB. Its size is {file_size / (1024 * 1024):.2f}MB.",
            UserWarning,
        )
        if chunk:
            chunk_file_with_split(filename, chunk_size="40M")
            if remove_unchunked:
                os.remove(filename)


def load_npz(filename):
    data = np.load(filename)
    return {k: v for k, v in data.items()}


def round_to(x, n=3):
    if x == 0:
        return 0.0
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def round_sci(x, n=3):
    return np.format_float_scientific(x, precision=n)


def log_log_fit(X, Y):
    """
    Computes linear best fit for log(X)-log(Y)
    """
    # warnings.warn(
    #     "returned dict keys will be updated in a future version. "
    #     "m->slope, b->intercept, F->fit_samples, fun->fit_fun. ",
    #     DeprecationWarning,
    #     stacklevel=2,
    # )
    logX, logY = np.log(X), np.log(Y)
    a11 = logX @ logX
    a12 = sum(logX)
    a21 = a12
    a22 = len(logX)
    u1 = logX @ logY
    u2 = sum(logY)
    u = np.array([u1, u2])
    detA = a11 * a22 - a12 * a21
    Ainv = np.array([[a22, -a12], [-a21, a11]]) / detA
    m, b = Ainv @ u
    F = m * logX + b
    fun = lambda x: np.exp(b) * x**m
    return {
        # "m": m,
        # "b": b,
        # "F": F,
        "logX": logX,
        "logY": logY,
        # "fun": fun,
        "slope": m,
        "intercept": b,
        "fit_samples": F,
        "fit_fun": fun,
    }


#############################################
# numdiff
def fib_disc(Npoints=100):
    xy = np.zeros((Npoints, 2))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        rad = np.sqrt(i / (Npoints - 1))  # radius at z

        theta = ga * i  # angle increment
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xy[i] = np.array([x, y])

    return xy


def fib_sphere(Npoints=100):
    xyz = np.zeros((Npoints, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        z = 1.0 - 2 * i / (Npoints - 1)  # -1<=z<=1
        rad = np.sqrt(1.0 - z**2)  # radius at z

        theta = ga * i  # angle increment

        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xyz[i] = np.array([x, y, z])

    return xyz


#############################
def matrix_to_quaternion(Q):
    """Converts a 3-by-3 rotation matrix to a unit quaternion. Safe to use with
    arbitrary rotation angle"""
    q = np.zeros(4)
    diagQ = np.array([Q[0, 0], Q[1, 1], Q[2, 2]])
    trQ = Q[0, 0] + Q[1, 1] + Q[2, 2]
    diagQmax = max(diagQ)

    use_alt_form = diagQmax > trQ
    if use_alt_form:
        # i = index_of(diagQ, diagQmax)
        i = list(diagQ).index(diagQmax)
        j = (i + 1) % 3  # index of the next elements
        k = (j + 1) % 3  # index of the next next element

        # multipliying by np.sign(Qkj_Qjk) ensures qs >=0
        qi = np.sqrt(1 + 2 * diagQmax - trQ) / 2
        Qkj_Qjk = Q[k, j] - Q[j, k]
        if Qkj_Qjk < 0:
            qi *= -1
        qj = (Q[i, j] + Q[j, i]) / (4 * qi)
        qk = (Q[i, k] + Q[k, i]) / (4 * qi)
        qs = Qkj_Qjk / (4 * qi)
    else:
        i, j, k = 0, 1, 2
        qs = np.sqrt(1 + trQ) / 2
        # cos_theta = 2 * qw**2 - 1
        qi = (Q[2, 1] - Q[1, 2]) / (4 * qs)
        qj = (Q[0, 2] - Q[2, 0]) / (4 * qs)
        qk = (Q[1, 0] - Q[0, 1]) / (4 * qs)

    q[0] = qs
    q[i + 1] = qi
    q[j + 1] = qj
    q[k + 1] = qk
    return q


def quaternion_to_matrix(q):
    qw, qx, qy, qz = q
    # qhat = np.array([[0.0, -qz, qy], [qz, 0.0, -qx], [-qy, qx, 0.0]])
    # qvec = np.array([qx, qy, qz])
    # I = np.eye(3)
    # q_qT = np.array(
    #     [
    #         [qx * qx, qx * qy, qx * qz],
    #         [qy * qx, qy * qy, qy * qz],
    #         [qz * qx, qz * qy, qz * qz],
    #     ]
    # )
    # Q = (qw**2 - qvec @ qvec) * I + 2 * qw * qhat + 2 * q_qT
    Q = np.array(
        [
            [
                qw**2 + qx**2 - qy**2 - qz**2,
                2 * qx * qy - 2 * qw * qz,
                2 * qw * qy + 2 * qx * qz,
            ],
            [
                2 * qw * qz + 2 * qx * qy,
                qw**2 - qx**2 + qy**2 - qz**2,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qw * qx + 2 * qy * qz,
                qw**2 - qx**2 - qy**2 + qz**2,
            ],
        ]
    )
    return Q
