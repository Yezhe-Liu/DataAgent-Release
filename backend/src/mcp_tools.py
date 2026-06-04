import asyncio
import json
import os
from typing import Any


def _parse_mcp_args(raw_args: str) -> list[str]:
    value = (raw_args or "").strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_allowlist(raw_allowlist: str) -> set[str]:
    return {item.strip() for item in (raw_allowlist or "").split(",") if item.strip()}


def _build_mcp_server_config() -> tuple[str | None, dict[str, Any], str]:
    server_name = os.getenv("MCP_SERVER_NAME", "primary").strip() or "primary"
    transport = os.getenv("MCP_SERVER_TRANSPORT", "streamable_http").strip().lower()

    if transport == "stdio":
        command = (os.getenv("MCP_SERVER_COMMAND", "")).strip()
        args = _parse_mcp_args(os.getenv("MCP_SERVER_ARGS", ""))
        if not command:
            return None, {}, "MCP_SERVER_COMMAND is required when transport=stdio"
        return (
            server_name,
            {
                "transport": "stdio",
                "command": command,
                "args": args,
            },
            "",
        )

    if transport in {"sse", "streamable_http"}:
        url = (os.getenv("MCP_SERVER_URL", "")).strip()
        if not url:
            return None, {}, "MCP_SERVER_URL is required when transport is sse/streamable_http"
        return (
            server_name,
            {
                "transport": transport,
                "url": url,
            },
            "",
        )

    return None, {}, f"Unsupported MCP transport: {transport}"


async def _load_mcp_tools_async() -> tuple[list[Any], str]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as error:
        return [], f"langchain-mcp-adapters import failed: {error}"

    server_name, server_config, config_error = _build_mcp_server_config()
    if config_error:
        return [], config_error

    try:
        client = MultiServerMCPClient({server_name: server_config})
        tools = await client.get_tools()
        allowlist = _parse_allowlist(os.getenv("MCP_TOOL_ALLOWLIST", ""))
        if allowlist:
            filtered_tools = [tool for tool in tools if getattr(tool, "name", "") in allowlist]
            return filtered_tools, ""
        return tools, ""
    except Exception as error:
        return [], f"MCP tool load failed: {error}"


def load_mcp_tools(local_tools: list[Any]) -> tuple[list[Any], str]:
    enabled = os.getenv("MCP_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
    mode = os.getenv("MCP_TOOL_MODE", "append").strip().lower()

    if not enabled:
        return list(local_tools), "disabled"

    if mode not in {"append", "replace"}:
        mode = "append"

    try:
        mcp_tools, err = asyncio.run(_load_mcp_tools_async())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            mcp_tools, err = loop.run_until_complete(_load_mcp_tools_async())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    if err:
        return list(local_tools), f"fallback_local: {err}"

    if mode == "replace":
        return mcp_tools, f"enabled_replace:{len(mcp_tools)}"

    return [*local_tools, *mcp_tools], f"enabled_append:{len(mcp_tools)}"
