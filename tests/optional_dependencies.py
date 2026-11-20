"""Skip tests whose optional packages the Python 3.12 workplace profile omits.

The research environment installs every package, so nothing is skipped there.
Set REQUIRE_OPTIONAL_TEST_DEPENDENCIES=1 (as the 3.14 CI job does) to turn a
missing package into an import failure instead of a skip.
"""

import importlib
import importlib.util
import os
import unittest


def _missing(modules):
    return [name for name in modules if importlib.util.find_spec(name) is None]


def requires(*modules):
    """Decorate a test or class that needs the named importable packages."""
    missing = _missing(modules)
    if missing and os.environ.get("REQUIRE_OPTIONAL_TEST_DEPENDENCIES") == "1":
        raise ModuleNotFoundError(f"Required test dependencies are not installed: {', '.join(missing)}")
    return unittest.skipIf(bool(missing), f"Optional dependency not installed: {', '.join(missing)}")


def optional_module(name):
    """Import a package if available; tests that use it must also be decorated with requires()."""
    return importlib.import_module(name) if not _missing((name,)) else None


SAMPLER_PACKAGES = ("emcee", "arviz")
