"""Configuration loading utilities.

The pipeline is driven entirely by a YAML config file (config/default.yaml).
This module loads it into a lightweight, dot-accessible object and supports
overriding individual keys (e.g. from CLI flags like --data-root).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Dict-backed config object with attribute access and dotted-key overrides.

    Example
    -------
    >>> cfg = Config.load("config/default.yaml")
    >>> cfg.grid.rows
    8
    >>> cfg.set("dataset.root", "/some/path")
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    # -- construction -----------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "Config":
        path = str(Path(path).expanduser())
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self._data))

    # -- access -------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(f"No config key '{item}'") from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, item: str) -> Any:
        return self.__getattr__(item)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        if isinstance(node, dict):
            return Config(node)
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """Override a (possibly nested) key, e.g. cfg.set('dataset.root', '/x')."""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string. 'auto' prefers Apple MPS, then CUDA, then CPU."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def expand_path(path: str) -> str:
    return str(Path(os.path.expanduser(path)).resolve()) if path else path
