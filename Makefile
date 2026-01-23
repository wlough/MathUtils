# ---------- Makefile for pymathutils (uv-based, supports multi-venv container) ----------

# Prefer Python 3.13 for dev envs (VTK/PyVista wheels exist through cp313).
# Override explicitly:  make PYVER=3.12 dev-editable
# Or override interpreter path/name: make UVPY=/usr/bin/python3.13 dev-editable
PY_CANDIDATES := 3.13 3.12 3.11 3.10

# If user sets UVPY, use it directly. Else choose first available pythonX.Y.
UVPY ?= $(shell \
	for v in $(PY_CANDIDATES); do \
	  command -v python$$v >/dev/null 2>&1 && { echo python$$v; exit 0; }; \
	done; \
	echo python3.13 \
)

# If user sets PYVER, prefer python$(PYVER) (e.g., python3.12) over auto-detect.
ifdef PYVER
UVPY := python$(PYVER)
endif

VENV ?= .venvs/build
BIN  := $(VENV)/bin

DISTDIR := dist
PKGNAME := pymathutils

REPO_ROOT := $(abspath .)
ABSBIN    := $(abspath $(BIN))

# Absolute paths and uv helpers 
VENV_ABS := $(abspath $(VENV))
UVPIP    := VIRTUAL_ENV=$(VENV_ABS) uv pip
UVRUN    := VIRTUAL_ENV=$(VENV_ABS) uv run

TESTVENV     := .venvs/testpypi
TESTVENV_ABS := $(abspath $(TESTVENV))
TEST_UVPIP   := VIRTUAL_ENV=$(TESTVENV_ABS) uv pip
TEST_UVRUN   := VIRTUAL_ENV=$(TESTVENV_ABS) uv run

.PHONY: help venv dev-editable build audit repair upload-test upload \
        test-install-testpypi test-import clean clean-all show-python

help:
	@echo "Targets:"
	@echo "  venv                 - create venv at $(VENV) and install build tooling"
	@echo "  dev-editable         - editable install in $(VENV)"
	@echo "  build                - build sdist+wheel into dist/ and run twine check"
	@echo "  audit                - show wheel audit info (Linux)"
	@echo "  repair               - repair Linux wheel to manylinux (uses auditwheel)"
	@echo "  upload-test          - upload dist/* to TestPyPI"
	@echo "  upload               - upload dist/* to PyPI"
	@echo "  test-install-testpypi- install from TestPyPI into $(TESTVENV)"
	@echo "  test-import          - smoke import with venv Python (avoids repo shadowing)"
	@echo "  clean                - remove dist/, build/, *.egg-info"
	@echo "  clean-all            - also remove venvs under .venvs/"
	@echo ""
	@echo "Python selection:"
	@echo "  make dev-editable                (auto-picks first found: $(PY_CANDIDATES))"
	@echo "  make PYVER=3.12 dev-editable     (force python3.12)"
	@echo "  make UVPY=/path/to/python dev-editable (force exact interpreter)"

show-python:
	@echo "UVPY=$(UVPY)"

# Create the venv using uv (no activation needed)
$(VENV):
	uv venv $(VENV) --python $(UVPY)

venv: $(VENV)
	$(UVPIP) install -U build twine

dev-editable: $(VENV)
	# Toolchain deps for scikit-build-core/pybind11 builds
	$(UVPIP) install -U scikit-build-core pybind11 cmake ninja
	# Editable install of this project (+ dev extras, if defined)
	$(UVPIP) install -U ".[dev]"
	$(UVPIP) install -e . --no-build-isolation --force-reinstall --no-deps -v
# 	$(UVPIP) install -e ".[dev]" --no-build-isolation --upgrade --force-reinstall --no-deps -v
# 	$(UVPIP) install -e ".[dev]" --no-build-isolation

build: venv
	rm -rf $(DISTDIR) build *.egg-info
	$(UVRUN) python -m build
	$(UVRUN) python -m twine check $(DISTDIR)/*

audit: venv
	$(UVPIP) install -U auditwheel
	$(UVRUN) auditwheel show $(DISTDIR)/*.whl || true

# Only attempt to repair plain linux-tagged wheels; ignore if none exist
repair: audit
	$(UVRUN) auditwheel repair -w $(DISTDIR) $(DISTDIR)/*-linux_*.whl || true
	find $(DISTDIR) -maxdepth 1 -type f -name "*linux_*.whl" ! -name "*manylinux*" -delete

upload-test: venv
	$(UVRUN) twine upload --verbose --repository test-pymathutils $(DISTDIR)/*

upload: venv
	$(UVRUN) twine upload --verbose --repository pymathutils $(DISTDIR)/*

# ---------- quick tests ----------
# Install from TestPyPI (with PyPI as fallback for deps) into a throwaway venv
test-install-testpypi:
	uv venv $(TESTVENV) --python $(UVPY)
	$(TEST_UVPIP) install \
	  -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple \
	  $(PKGNAME)==$$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

test-import: $(VENV)
	"$(ABSBIN)/python" -c "import os, sys; \
	  (sys.path.pop(0) if os.path.abspath(sys.path[0]) == os.path.abspath('$(REPO_ROOT)') else None); \
	  import $(PKGNAME) as m; from $(PKGNAME) import mathutils_backend as be; \
	  print('OK import:', m.__version__, '| backend:', be.__name__)"

clean:
	rm -rf $(DISTDIR) build *.egg-info .pytest_cache __pycache__

clean-all: clean
	rm -rf .venvs
