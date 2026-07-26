"""Deterministic cache keys for slow validation solves.

Validation caches are evidence, not production inputs. Reusing one after the
solver, gate geometry, or numerical settings change can hide a regression, so
the filename binds to both the complete physics key and the source files that
produce the result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from glob import glob
from typing import Any, Iterable

import numpy as np
import scipy


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def cache_path(prefix: str, physics_key: Any,
               source_paths: Iterable[str]) -> str:
    """Return ``.<prefix>_<sha256[:16]>.pkl`` in the current directory."""

    name = str(prefix).strip().replace(os.sep, "_")
    if not name:
        raise ValueError("validation cache prefix cannot be empty.")
    digest = hashlib.sha256()
    digest.update(b"validation-cache-v1\0")
    digest.update(json.dumps({
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(
        _jsonable(physics_key), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))
    for path in sorted(os.path.abspath(str(item)) for item in source_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"validation cache source does not exist: {path}")
        digest.update(b"\0" + path.encode("utf-8") + b"\0")
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return f".{name}_{digest.hexdigest()[:16]}.pkl"


def bor_solver_sources(backend_dir: str) -> tuple[str, ...]:
    """All local implementation files that can change a BoR field solve."""

    backend = os.path.abspath(str(backend_dir))
    sources = [
        os.path.join(backend, "bor_solver.py"),
        os.path.join(backend, "bor_kernels.py"),
        os.path.join(backend, "bor_streaming.py"),
        os.path.join(backend, "bor_stream_kernel.c"),
    ]
    sources.extend(sorted(glob(os.path.join(
        backend, "bor_stream_kernel*.so"))))
    return tuple(path for path in sources if os.path.isfile(path))
