"""Registry for matching strategies."""

from __future__ import annotations

from typing import Callable

Matcher = Callable[[list[dict]], list[tuple[dict, dict]]]


MATCHERS: dict[str, Matcher] = {}


def register(name: str, matcher: Matcher) -> None:
    MATCHERS[name] = matcher


def get(name: str) -> Matcher:
    return MATCHERS[name]
