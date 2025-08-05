def generate_numpyarray(name, nvalues, dtype="float64", numpy_name="np"):
    indent = 4 * " "
    # new_line = f",\n{2*pad}"
    # template = """{name} = np.array(\n{pad}[\n{pad}{pad}{values},\n{pad}],\n{pad}dtype="float64",\n)"""
    code = f"{name} = {numpy_name}.array(\n"
    code += f"{indent}[\n"
    code += f"{2*indent}"
    code += f",\n{2*indent}".join([f"{_:.15g}" for _ in nvalues]) + ",\n"
    code += f"{indent}],\n"
    code += f"""{indent}dtype="float64",\n"""
    code += ")"
    return code


def generate_carray(name, nvalues, dtype="double"):
    code = f"{dtype} {name}[{len(nvalues)}] ={{"
    indent = len(code) * " "
    print(indent)
    code += f",\n{indent}".join([f"{_:.15g}" for _ in nvalues]) + "};"
    return code


def generate_carray2d(name, nvalues, dtype="double"):
    code = f"{dtype} {name}[{len(nvalues)}][{len(nvalues[0])}] ={{"
    indent = len(code) * " "
    code += (
        f",\n{indent}".join(
            [f"{{{', '.join([f'{_:.15g}' for _ in row])}}}" for row in nvalues]
        )
        + "};"
    )
    return code


def generate_constexpr_carray(name, nvalues, dtype="double"):
    code = generate_carray(name, nvalues, dtype=dtype)
    prefix = "constexpr "
    indent = len(prefix) * " "
    code = indent_code(code, indent)
    code = prefix + code[len(prefix) :]
    return code


def generate_constexpr_carray2d(name, nvalues, dtype="double"):
    code = generate_carray2d(name, nvalues, dtype=dtype)
    prefix = "constexpr "
    indent = len(prefix) * " "
    code = indent_code(code, indent)
    code = prefix + code[len(prefix) :]
    return code


def indent_code(code, indent=4 * " "):
    return indent + f"\n{indent}".join(code.split("\n"))


def generate_high_precision_log_factorial_table(
    n_max=200, precision=200, code_type="cpp"
):
    """Generate high-precision log factorial table"""
    from decimal import Decimal, getcontext

    getcontext().prec = precision

    nvalues = []
    for n in range(n_max + 1):
        if n <= 1:
            nvalues.append(0.0)
        else:
            log_fact = sum(Decimal(str(i)).ln() for i in range(2, n + 1))
            nvalues.append(float(log_fact))
    if code_type == "cpp":
        code = generate_constexpr_carray(
            "LOG_FACTORIAL_LOOKUP_TABLE",
            nvalues,
            dtype="double",
        )

    if code_type == "np":
        code = generate_numpyarray(
            "LOG_FACTORIAL_LOOKUP_TABLE",
            nvalues,
            dtype="float64",
            numpy_name="np",
        )

    return code, nvalues


def generate_spherical_harmonics_index_table(l_max=200):
    """Generate spherical harmonics index table"""
    import numpy as np

    # n_max = l_max * (l_max + 2)
    # n_LM = m+l*(l+1)
    # lm_N = [floor(sqrt(n)), n - floor(sqrt(n))*(floor(sqrt(n)) + 1)]
    # l = floor(sqrt(n))
    # m = n - l * (l + 1)
    lm_N = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            lm_N.append([l, m])

    code = generate_constexpr_carray2d(
        "SPHERICAL_HARMONIC_INDEX_LM_N_LOOKUP_TABLE",
        lm_N,
        dtype="int",
    )

    return code


def write_code(code_str, save_path):
    """Write code stored as str to a file"""
    with open(save_path, "w") as f:
        f.write(code_str)
    print(f"Code written to {save_path}")


def write_log_factorial_lookup_table(n_max=200, precision=200):
    """Generate and write lookup tables for mathutils"""

    look_up_tables_hpp = "#pragma once\n\n"
    look_up_tables_hpp += "#include <cstdint>\n\n"
    look_up_tables_hpp += (
        "/**\n"
        "* @file log_factorial_lookup_table.hpp\n"
        r"* @brief table of \f$\ln(n!)\f$ for \f$n \in [0, {n_max}]\f$" + "\n"
        "*/\n\n"
    )
    look_up_tables_hpp += "namespace mathutils {\n\n"
    look_up_tables_hpp += (
        f"constexpr uint64_t LOG_FACTORIAL_LOOKUP_TABLE_SIZE = {n_max+1};\n\n"
    )
    log_factorial_table, _ = generate_high_precision_log_factorial_table(
        n_max=n_max, precision=precision, code_type="cpp"
    )
    look_up_tables_hpp += log_factorial_table + "\n\n"
    look_up_tables_hpp += "} // namespace mathutils\n"
    write_code(
        look_up_tables_hpp,
        save_path="./include/mathutils/log_factorial_lookup_table.hpp",
    )

    # look_up_tables_py = "# Generated lookup tables for mathutils\n"
    # look_up_tables_py += "import numpy as np\n\n"
    # log_factorial_table, _ = generate_high_precision_log_factorial_table(
    #     n_max=50, precision=50, code_type="np"
    # )
    # look_up_tables_py += log_factorial_table + "\n\n"
    # look_up_tables_py += "__all__ = [\n"
    # look_up_tables_py += "    'LOG_FACTORIAL_LOOKUP_TABLE',\n"
    # look_up_tables_py += "]\n"
    # write_code(look_up_tables_py, save_path="./mathutils/lookup_tables.py")


def write_spherical_harmonic_index_lookup_table(l_max=100):
    """Generate and write lookup tables for mathutils"""
    look_up_tables_hpp = "#pragma once\n\n"
    look_up_tables_hpp += (
        "/**\n"
        "* @file spherical_harmonic_index_lookup_table.hpp\n"
        "* @brief tables of ordered spherical harmonic indices\n"
        "*/\n\n"
    )
    look_up_tables_hpp += "namespace mathutils {\n\n"
    look_up_tables_hpp += f"constexpr int SPHERICAL_HARMONIC_INDEX_L_MAX = {
        l_max};\n\n"
    look_up_tables_hpp += f"constexpr int SPHERICAL_HARMONIC_INDEX_N_MAX = {
            l_max * (l_max + 2)};\n\n"
    arr = generate_spherical_harmonics_index_table(l_max=l_max)
    look_up_tables_hpp += arr + "\n\n"
    look_up_tables_hpp += "} // namespace mathutils\n"
    write_code(
        look_up_tables_hpp,
        save_path="./include/mathutils/spherical_harmonics_index_lookup_table.hpp",
    )
