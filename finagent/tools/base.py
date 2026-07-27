"""
base.py  ·  finagent/tools/base.py

Abstract interface for agent tools. A `BaseTool` has a name, a description
(shown to the LLM when selecting tools), and a `run(**kwargs)` method that
returns a plain dict. The graph nodes call each tool's `run` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Common interface for agent tools.

    Subclasses set `name` and `description` (class attributes) and implement
    `run`. Keep `run` returning a plain ``dict`` so results serialize cleanly
    into agent state and SSE events.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> dict:
        """Execute the tool and return a JSON-serializable result dict."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
