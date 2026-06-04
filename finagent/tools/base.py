"""
base.py  ·  finagent/tools/base.py

Abstract interface for agent tools plus a small registry. A `BaseTool` has a
name, a description (shown to the LLM when selecting tools), and a `run(**kwargs)`
method. `as_langchain_tool()` adapts it to a LangChain `Tool`, and `ToolRegistry`
collects tools so the agent can hand the LLM a uniform list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


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

    def as_langchain_tool(self):
        """Adapt this tool to a LangChain `Tool` wrapping `run()`."""
        from langchain_core.tools import Tool

        return Tool(name=self.name, description=self.description,
                    func=lambda **kw: self.run(**kw))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class ToolRegistry:
    """A name→tool registry the agent can query for available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add a tool (keyed by its `name`)."""
        if not tool.name:
            raise ValueError(f"{tool!r} has no name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_langchain_tools(self) -> list:
        """All registered tools as LangChain `Tool` objects."""
        return [t.as_langchain_tool() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
