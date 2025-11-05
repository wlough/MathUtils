# ---------- Makefile for pymathutils ----------
# Use '>' instead of a tab to start recipe lines (GNU Make ≥ 3.82)
.RECIPEPREFIX := >

PY      ?= python
VENV    ?= .venv/build
BIN     := $(VENV)/bin
DISTDIR := dist
PKGNAME := pymathutils

# Absolute paths (robust even if a recipe changes cwd)
REPO_ROOT := $(abspath .)
ABSBIN    := $(abspath $(BIN))

.PHONY: help venv dev-editable build audit repair upload-test upload test-import clean clean-all

help:
> @echo "Targets:"
> @echo "  dev-editable  - editable install in .venv/build"
> @echo "  build         - build sdist+wheel into dist/ and run twine check"
> @echo "  audit         - show wheel audit info (Linux)"
> @echo "  repair        - repair Linux wheel to manylinux (uses auditwheel)"
> @echo "  upload-test   - upload dist/* to TestPyPI"
> @echo "  upload        - upload dist/* to PyPI"
> @echo "  test-import   - smoke import with venv Python (avoids repo shadowing)"
> @echo "  clean         - remove dist/, build/, *.egg-info"
> @echo "  clean-all     - also remove .venv/build"

$(VENV):
> $(PY) -m venv $(VENV)
> $(BIN)/python -m pip install -U pip

venv: $(VENV)
> $(BIN)/python -m pip install -U build twine

dev-editable: $(VENV)
> $(BIN)/python -m pip install -U pip
> $(BIN)/python -m pip install -U scikit-build-core pybind11 cmake ninja
> $(BIN)/python -m pip install -e . --no-build-isolation

build: venv
> rm -rf $(DISTDIR) build *.egg-info
> $(BIN)/python -m build
> $(BIN)/python -m twine check $(DISTDIR)/*

audit: venv
> $(BIN)/python -m pip install -U auditwheel
> $(BIN)/auditwheel show $(DISTDIR)/*.whl || true

# Only attempt to repair plain linux-tagged wheels; ignore if none exist
repair: audit
> $(BIN)/auditwheel repair -w $(DISTDIR) $(DISTDIR)/*-linux_*.whl || true
> find $(DISTDIR) -maxdepth 1 -type f -name "*linux_*.whl" ! -name "*manylinux*" -delete

upload-test: venv
> $(BIN)/python -m twine upload --verbose --repository testpypi $(DISTDIR)/*

upload: venv
> $(BIN)/python -m twine upload --verbose $(DISTDIR)/*

# ---------- quick tests ----------
# Install from TestPyPI (with PyPI as fallback for deps) into a throwaway venv
test-install-testpypi:
> $(PY) -m venv .venv/testpypi
> . .venv/testpypi/bin/activate; \
> python -m pip install -U pip; \
> python -m pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple $(PKGNAME)==$$(python -c "import tomllib,sys;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

test-import: $(VENV)
> "$(ABSBIN)/python" -c "import os, sys; \
>   (sys.path.pop(0) if os.path.abspath(sys.path[0]) == os.path.abspath('$(REPO_ROOT)') else None); \
>   import $(PKGNAME) as m; from $(PKGNAME) import mathutils_backend as be; \
>   print('OK import:', m.__version__, '| backend:', be.__name__)"


clean:
> rm -rf $(DISTDIR) build *.egg-info .pytest_cache __pycache__

clean-all: clean
> rm -rf $(VENV) .venv/testpypi .venv/wheel
