"""Tiny memoization helper built on :func:`functools.lru_cache`.

`df_memoize` wraps a function whose inputs may include :class:`pandas.DataFrame`
instances and other unhashable objects (dicts, etc). These are not hashable by
content, but in the parameter sweep they are held by stable references that are
never mutated in place, so identity (:func:`id`) is a sound cache key. The
:class:`_IdentityProxy` proxy below lets such objects participate in
`lru_cache`'s hashable key.

Caveats:
- Callers MUST NOT mutate the input DataFrames between calls. The sweep holds
  signal/prices/transactions frames alive and unchanged for its entire run.
- By default DataFrame RETURN values are handed back as fresh copies so callers
  can mutate them (insert columns, .loc assignments) without corrupting the
  cache. Pass ``copy=False`` for "structural" helpers whose canonical cached
  object must keep a stable id() across calls so downstream memoized functions
  can key on it (filter helpers, rank_members, ...). When ``copy=False`` the
  cached DataFrame itself is returned and callers MUST treat it as read-only.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import pandas as pd


class _IdentityProxy:
    """Identity-based hashable proxy for unhashable objects (DataFrame, dict, etc)."""

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _IdentityProxy) and other.obj is self.obj


def _wrap(x: Any) -> Any:
    if isinstance(x, pd.DataFrame):
        return _IdentityProxy(x)
    if isinstance(x, dict):
        return _IdentityProxy(x)
    return x


def _unwrap(x: Any) -> Any:
    return x.obj if isinstance(x, _IdentityProxy) else x


def df_memoize(func: Callable | None = None, *, copy: bool = True) -> Callable:
    """Memoize ``func`` via :func:`functools.lru_cache`, keyed on arg identity.

    DataFrame arguments are wrapped in :class:`_DF` so they hash by :func:`id`.
    When ``copy`` is True (default), DataFrame return values are copied before
    being returned to the caller so the cache is safe against caller mutation.
    When ``copy`` is False, the cached DataFrame is returned directly so its
    id() stays stable across calls — use this only for read-only shared inputs
    to other memoized functions (filter helpers, rank_members, ...).
    """

    def _decorator(f: Callable) -> Callable:
        @functools.lru_cache(maxsize=None)
        def _cached(*w_args: Any, **w_kwargs: Any) -> Any:
            args = tuple(_unwrap(a) for a in w_args)
            kwargs = {k: _unwrap(v) for k, v in w_kwargs.items()}
            return f(*args, **kwargs)

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            w_args = tuple(_wrap(a) for a in args)
            w_kwargs = {k: _wrap(v) for k, v in kwargs.items()}
            result = _cached(*w_args, **w_kwargs)
            if copy and isinstance(result, pd.DataFrame):
                return result.copy()
            return result

        wrapper.cache_clear = _cached.cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = _cached.cache_info  # type: ignore[attr-defined]
        return wrapper

    if func is None:
        return _decorator
    return _decorator(func)
