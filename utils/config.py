"""Configuration loading and path resolution.

IMPLEMENTATION DECISION: configuration is JSON (stdlib only) so the pipeline has
no configuration-format dependency. All paths in the config are interpreted
relative to the project root, never to the current working directory, so there
are no machine-specific absolute paths anywhere in the repository.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def portable_path(value: str | Path) -> str:
    """Render a path for storage in a metrics file.

    Absolute paths are made relative to the project root and POSIX-separated.
    Metrics files are committed and published, so an absolute path would leak
    the machine's directory layout (and typically the user's name) into the
    public record while adding nothing reproducible.
    """
    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Outside the project tree: keep only the final component.
        return path.name


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a JSON config, optionally applying dotted-key overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = resolve_path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = portable_path(cfg_path)
    for key, value in (overrides or {}).items():
        set_nested(cfg, key, value)
    return cfg


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = value`` from the key ``'a.b'``."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key.path=value`` CLI override. Values are parsed as JSON when possible."""
    if "=" not in text:
        raise ValueError(f"Override must look like key.path=value, got: {text!r}")
    key, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def output_dir(cfg: dict, *parts: str) -> Path:
    """Return (and create) an output subdirectory."""
    path = resolve_path(cfg["paths"]["outputs"]).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    """Write a JSON document with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def config_snapshot(cfg: dict) -> dict:
    """A copy of the config safe to embed in a metrics file."""
    return copy.deepcopy(cfg)
