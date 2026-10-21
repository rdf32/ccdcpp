"""The Python tests' start-up, shared by every test file.

A test runs as `python test_<name>.py <module_dir> <source_dir> [unittest args]`:
<module_dir> holds the built extension (ccdcpp.*.so / .pyd), <source_dir> is the
repository root. Importing this module reads the two and removes them from sys.argv,
once however many modules import it. A test then imports `ccdcpp` as one of:

  * the package, laid out as the wheel lays it out (python/ccdcpp with its configs, the
    extension beside them) in <module_dir>/test_package_<tag> (use_package()), one
    directory per test file so parallel runs do not share it;
  * the bare extension module of <module_dir> (use_extension()).
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

if len(sys.argv) < 3:
    sys.exit(f"usage: python {os.path.basename(sys.argv[0])} <module_dir> <source_dir> "
             "[unittest args]")

MODULE_DIR, SOURCE_DIR = sys.argv[1], sys.argv[2]
del sys.argv[1:3]

#: The running test file's name, without .py.
SCRIPT = os.path.splitext(os.path.basename(sys.argv[0]))[0]


def assemble(root: str) -> str:
    """The package laid out in `root` as the wheel lays it out: python/ccdcpp (its
    configs too) with the built extension beside it. Returns its __init__.py."""
    package = os.path.join(root, "ccdcpp")
    shutil.rmtree(package, ignore_errors=True)
    shutil.copytree(os.path.join(SOURCE_DIR, "python", "ccdcpp"), package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (extension,) = glob.glob(os.path.join(MODULE_DIR, "ccdcpp.*"))
    shutil.copy(extension, package)
    return os.path.join(package, "__init__.py")


def use_package(tag: str = SCRIPT) -> str:
    """Puts the package, assembled in <module_dir>/test_package_<tag>, first on
    sys.path, so `import ccdcpp` imports it. Returns its __init__.py, which the
    caller checks ccdcpp.__file__ against."""
    root = os.path.join(MODULE_DIR, "test_package_" + tag)
    shutil.rmtree(root, ignore_errors=True)
    init = assemble(root)
    sys.path.insert(0, root)
    return init


def use_extension() -> None:
    """Puts <module_dir> first on sys.path, so `import ccdcpp` imports the bare
    extension module, which python/ccdcpp/ccdcpp.pyi types."""
    sys.path.insert(0, MODULE_DIR)
