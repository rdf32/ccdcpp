"""The extension's type stubs (python/ccdcpp/ccdcpp.pyi, qa_tables.pyi) against the extension.

usage: python test_stub.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root. ctest runs this
as `python_stub`. Every name the extension module defines is in its stub and every name the
stub defines is in the extension; so are each class's members, each function's parameter names
(as the extension's signature line gives them) and Input's fields.
"""
from __future__ import annotations

import ast
import inspect
import os
import re
import unittest
from collections.abc import Callable, Sequence

from _harness import SOURCE_DIR, use_package

use_package()

# The extension module, which ccdcpp/ccdcpp.pyi types.
from ccdcpp import ccdcpp as ext  # noqa: E402

STUBS = os.path.join(SOURCE_DIR, "python", "ccdcpp")


def parse(name: str) -> ast.Module:
    with open(os.path.join(STUBS, name), encoding="utf-8") as f:
        return ast.parse(f.read())


def is_alias(node: ast.stmt) -> bool:
    """A stub-only type alias: `_Name: TypeAlias = ...`."""
    return (isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name)
            and node.annotation.id == "TypeAlias")


def defined(body: Sequence[ast.stmt]) -> dict[str, ast.stmt]:
    """{name: node} of what a module or class body defines: functions, classes, annotated or
    assigned names and explicit re-exports (`from . import x as x`), without type aliases."""
    out: dict[str, ast.stmt] = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and not is_alias(node):
            out[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname == alias.name:
                    out[alias.name] = node
    return out


def runtime_names(module: object) -> set[str]:
    """The module's names, the dunders other than __version__ left out."""
    return {n for n in dir(module)
            if not (n.startswith("__") and n.endswith("__")) or n == "__version__"}


def stub_names(body: Sequence[ast.stmt]) -> set[str]:
    return {n for n in defined(body)
            if not (n.startswith("__") and n.endswith("__")) or n == "__version__"}


def members(cls: type) -> set[str]:
    """A class's own public members."""
    return {n for n in vars(cls) if not n.startswith("_")}


def doc_parameters(function: Callable[..., object]) -> list[str]:
    """The parameter names of a pybind11 function's first signature line."""
    first = (function.__doc__ or "").lstrip().split("\n", 1)[0]
    return re.findall(r"(?:\(|, )(?:\*, )?(\w+):", first)


def stub_parameters(node: ast.FunctionDef) -> list[str]:
    args = node.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


class TestStub(unittest.TestCase):

    def test_module_names(self):
        stub = stub_names(parse("ccdcpp.pyi").body)
        runtime = runtime_names(ext)
        self.assertEqual(sorted(runtime - stub), [], "in the extension, not in ccdcpp.pyi")
        self.assertEqual(sorted(stub - runtime), [], "in ccdcpp.pyi, not in the extension")

    def test_qa_tables_names(self):
        stub = stub_names(parse("qa_tables.pyi").body)
        runtime = runtime_names(ext.qa_tables)
        self.assertEqual(sorted(runtime - stub), [], "in qa_tables, not in qa_tables.pyi")
        self.assertEqual(sorted(stub - runtime), [], "in qa_tables.pyi, not in qa_tables")

    def test_class_members(self):
        for name, node in defined(parse("ccdcpp.pyi").body).items():
            if not isinstance(node, ast.ClassDef):
                continue
            with self.subTest(cls=name):
                stub = {n for n in defined(node.body) if not n.startswith("_")}
                self.assertEqual(stub, members(getattr(ext, name)))

    def test_parameters(self):
        modules = (("ccdcpp.pyi", ext), ("qa_tables.pyi", ext.qa_tables))
        for stub, module in modules:
            body = defined(parse(stub).body)
            for name, node in body.items():
                if isinstance(node, ast.FunctionDef):
                    with self.subTest(function=name):
                        self.assertEqual(stub_parameters(node),
                                         doc_parameters(getattr(module, name)))
                if isinstance(node, ast.ClassDef):
                    cls = getattr(module, name)
                    for method, m in defined(node.body).items():
                        if not isinstance(m, ast.FunctionDef) or method.startswith("_") \
                                or any(getattr(d, "id", "") == "property"
                                       for d in m.decorator_list):
                            continue
                        if method == "replace":
                            continue   # **fields, as Input(); test_input_fields checks them
                        with self.subTest(method=f"{name}.{method}"):
                            self.assertEqual(stub_parameters(m),
                                             doc_parameters(getattr(cls, method)))

    def test_input_fields(self):
        # Input(**fields) and replace(**fields) take every field the Input has.
        fields = {n for n, v in vars(ext.Input).items()
                  if not n.startswith("_") and not inspect.isroutine(v)}
        (node,) = [n for n in parse("ccdcpp.pyi").body
                   if isinstance(n, ast.ClassDef) and n.name == "Input"]
        for method in ("__init__", "replace"):
            with self.subTest(method=method):
                (m,) = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method]
                self.assertEqual({a.arg for a in m.args.kwonlyargs}, fields)
                self.assertEqual(m.args.args[1:], [])


if __name__ == "__main__":
    unittest.main()
