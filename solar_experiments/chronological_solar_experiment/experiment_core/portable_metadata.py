"""Portable JSON: paths are relative to the containing file, never a machine."""
from __future__ import annotations

import json
import os
from pathlib import Path


MACHINE_FIELDS = frozenset({
    "hostname", "machine", "platform", "processor", "system", "environment",
    "python_version", "numpy_version", "torch_version", "cuda_version",
    "device", "device_name", "executable", "cwd", "command_line",
})


def portable_values(value, base_dir):
    """Convert runtime paths to file-relative references before serialization."""
    base_dir = Path(base_dir).resolve()
    if isinstance(value, dict):
        return {
            str(key): portable_values(item, base_dir)
            for key, item in value.items()
            if str(key) not in MACHINE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [portable_values(item, base_dir) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(os.path.relpath(value, base_dir)).as_posix()
    return value


def save_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(portable_values(payload, path.parent), handle, indent=2)
        handle.write("\n")


def runtime_paths(value, base_dir):
    """Resolve saved file/directory fields when reading a relocated run."""
    if isinstance(value, list):
        return [runtime_paths(item, base_dir) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if (
            isinstance(item, str)
            and (key == "path" or key.endswith(("_path", "_dir", "_root")))
        ):
            result[key] = str((Path(base_dir) / item).resolve())
        else:
            result[key] = runtime_paths(item, base_dir)
    return result
