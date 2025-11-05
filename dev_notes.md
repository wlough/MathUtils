
# Development & Release notes

This repo ships a `Makefile` to automate dev installs, builds, wheel repair, and uploads. You **don’t need to activate a venv**—targets use an internal build venv at `.venv/build`.

---

## Prereqs

* Python ≥ 3.8, `g++`, `make`
* Optional (Linux wheel repair): `auditwheel` (installed automatically by `make repair`)
* Optional: `~/.pypirc` configured for `pypi` and `testpypi`

---

## Common Tasks

```bash
# 1) Editable dev install (builds C++ once, Python live-reloads)
make dev-editable

# 2) Quick smoke test (imports with current interpreter)
make test-import

# 3) Build artifacts (sdist + wheel) and run Twine checks
make build

# 4) Linux only: repair wheel → manylinux (required for PyPI/TestPyPI)
make repair

# 5) Upload to TestPyPI (dry run)
make upload-test

# 6) Upload to PyPI (real release)
make upload

# 7) Clean artifacts / venvs
make clean        # dist/, build/, *.egg-info, caches
make clean-all    # plus removes .venv/build and test venvs
```

---

## Full Release Flow

```bash
# Dev loop
make dev-editable
make test-import

# Build + validate
make build

# Linux wheel: repair to manylinux
make repair

# Dry-run in TestPyPI
make upload-test

# Install from TestPyPI in a throwaway venv and smoke test
make test-install-testpypi

# Publish to PyPI
make upload
```

---

## Notes & Tips

* Bump the version in `pyproject.toml` before building; PyPI/TestPyPI reject re-uploads.
* The backend builds a module installed as `pymathutils.mathutils_backend`.
* If you changed C++ code, re-run `make dev-editable` to rebuild. Python-only changes require no rebuild.
* Want an interactive shell in the build venv?
  `source .venv/build/bin/activate` (optional; Makefile doesn’t require it).
* On Linux, **you must upload a repaired `manylinux_*` wheel** or just upload the sdist.
* `make -n dev-editable` to print the commands Make would run without executing them













# Create build venv
python -m venv .venv/mathutils_build
source .venv/mathutils_build/bin/activate

python -m pip install -U pip build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build sdist + wheel (scikit-build-core drives CMake)
python -m build

# Inspect artifacts
ls -la dist/
python -m twine check dist/*

# (Optional but recommended) TestPyPI first
# python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*



# For old version with setup.py
# Create build venv
python -m venv .venv/mathutils_build
source .venv/mathutils_build/bin/activate
pip install build twine
# Clean previous builds
rm -rf dist/ build/ *.egg-info
# Build source distribution and wheel
python -m build
# Verify files were created
ls -la dist/
# Check package validity
python -m twine check dist/*
# Upload only .tar.gz to avoid platform-specific wheel issues
python -m twine upload dist/*.tar.gz
# test
deactivate
python -m venv .venv/mathutils_test
source .venv/mathutils_test/bin/activate
pip install -r requirements_testing.txt









# 0) Start in your repo root (where pyproject.toml lives)
# 1) Create & activate a clean venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
# 2) Editable dev install (fast iteration)
python -m pip install -e .
# 3) Sanity check imports from OUTSIDE your repo (avoid shadowing by ./pymathutils)
cd ..
python -c "import pymathutils, pymathutils.mathutils_backend as be; print(pymathutils.__version__)"
# 4) Come back for dev work
cd -    # back to repo
# Rebuilding after changes
python -m pip install -e .      # rebuilds the compiled module



# Wheel & sdist validation (what end-users get)

# Build artifacts
python -m pip install -U build
python -m build           # creates dist/*.whl and dist/*.tar.gz
# Check metadata
python -m pip install -U twine
python -m twine check dist/*
# Fresh venv to test the wheel like a user
deactivate
python -m venv .venv-wheel
source .venv-wheel/bin/activate
python -m pip install dist/*.whl
# Import test from a neutral directory
cd ..
python -c "import pymathutils, pymathutils.mathutils_backend as be; print('OK', pymathutils.__version__)"



# Clean local build junk
rm -rf build/ dist/ *.egg-info *.egg .pytest_cache