"""The comparison of entry point outputs the Python tests share."""
from __future__ import annotations

import unittest
from typing import Any

import numpy as np


def assert_same(test: unittest.TestCase, a: Any, b: Any, where: str = "out") -> None:
    """Exact, structural equality of entry point outputs."""
    if isinstance(a, dict):
        test.assertEqual(list(a), list(b), where)
        for k in a:
            assert_same(test, a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(type(a), type(b), where)
        test.assertEqual(len(a), len(b), where)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{where}[{i}]")
    elif isinstance(a, np.ndarray):
        test.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), where)
        test.assertEqual(a.tobytes(), b.tobytes(), where)
    elif isinstance(a, float):
        test.assertEqual(a.hex(), b.hex(), where)
    else:
        test.assertEqual(a, b, where)
