from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


class SeoTools:
    """
    Central registry for all MCP tools (GSC + DataForSEO).
    - Creates the MultiServerMCPClient once.
    - Loads tools once and caches them.
    - Provides helper methods to access tools when needed.
    """

    def __init__(self) -> None:
        load_dotenv()
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: Optional[List[Any]] = None
        self._tools_by_name: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # MCP client creation
    # -------------------------------------------------------------------------
    def _build_mcp_client(self) -> MultiServerMCPClient:
        current_dir = Path.cwd()

        gsc_server_path = current_dir / "src" / "tools" / "gsc_server.py"
        if not gsc_server_path.exists():
            print(f"Warning: Could not find gsc_server.py at: {gsc_server_path}")

        python_interpreter = sys.executable

        return MultiServerMCPClient(
            {
                "gscServer": {
                    "command": python_interpreter,
                    "args": [str(gsc_server_path)],
                    "transport": "stdio",
                    "env": {
                        "GSC_CREDENTIALS": os.getenv("GSC_CREDENTIALS", ""),
                        "GSC_SKIP_OAUTH": os.getenv("GSC_SKIP_OAUTH", "true"),
                    },
                },
                "dataforseo": {
                    "transport": "streamable_http",
                    "url": "https://dataforseo-mcp-worker.hitesh-solanki.workers.dev/mcp",
                },
            }
        )

    @property
    def client(self) -> MultiServerMCPClient:
        """Lazily create the MCP client once."""
        if self._client is None:
            self._client = self._build_mcp_client()
        return self._client

    # -------------------------------------------------------------------------
    # Tool loading & indexing
    # -------------------------------------------------------------------------
    async def _load_tools(self, force_reload: bool = False) -> List[Any]:
        """
        Load tools from all MCP servers via MultiServerMCPClient.
        Caches the result unless force_reload=True.
        """
        if self._tools is None or force_reload:
            self._tools = await self.client.get_tools()
            self._index_tools()
        return self._tools

    def _index_tools(self) -> None:
        """Build simple indexes (currently: by name)."""
        self._tools_by_name.clear()
        if not self._tools:
            return

        for tool in self._tools:
            # LangChain tools usually have .name
            name = getattr(tool, "name", None)
            if name:
                self._tools_by_name[name] = tool

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    async def get_all_tools(self, force_reload: bool = False) -> List[Any]:
        """Get the full list of tools (GSC + DataForSEO)."""
        return await self._load_tools(force_reload=force_reload)

    async def get_tool_by_name(self, name: str) -> Optional[Any]:
        """Look up a single tool by exact name."""
        await self._load_tools()
        return self._tools_by_name.get(name)

    async def get_tools_by_prefix(self, prefix: str) -> List[Any]:
        """Convenience method - fetch tools whose name starts with a prefix."""
        await self._load_tools()
        return [
            tool
            for tool_name, tool in self._tools_by_name.items()
            if tool_name.startswith(prefix)
        ]

    async def get_tools_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dict mapping tool_name -> {description, args_schema}.
        """
        tools = await self.get_all_tools()
        meta: Dict[str, Dict[str, Any]] = {}

        for t in tools:
            name = getattr(t, "name", None)
            if not name:
                continue
            desc = getattr(t, "description", "") or ""
            args_schema = getattr(t, "args_schema", {}) or {}
            meta[name] = {
                "description": desc,
                "args_schema": args_schema,
            }

        return meta
    
    async def run_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with the provided args.

        This is what the generated code ultimately calls (through SEOCodeExecutor).
        """
        await self._load_tools()
        tool = self._tools_by_name.get(tool_name)

        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")

        # LangChain StructuredTool usually supports .ainvoke(input)
        # Here we pass the args dict as the single input.
        # If your adapter expects kwargs, you can switch to: await tool.arun(**args)
        return await tool.ainvoke(args)