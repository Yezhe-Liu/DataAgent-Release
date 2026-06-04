"""通用 MCP Server — 配置驱动多数据库查询

从 mcp_databases.json 读取数据库配置, 自动为每个数据库注册一个只读查询工具。
可适配企业政策、电信气象、合规管理等多种业务场景。

启动方式 (stdio):
    uv run python -m src.mcp_server
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = os.getenv("MCP_DB_CONFIG", str(BASE_DIR / "data" / "mcp_databases.json"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _DB_CONFIGS: list[dict[str, Any]] = json.load(f).get("databases", [])

# 工具名 → 数据库路径映射
_TOOL_DB_MAP: dict[str, str] = {
    cfg["tool_name"]: cfg["db_path"] for cfg in _DB_CONFIGS
}

# ---------------------------------------------------------------------------
# SQL 安全检查
# ---------------------------------------------------------------------------

_DANGEROUS_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|"
    r"EXEC|EXECUTE|MERGE|LOAD|INTO\s+(OUTFILE|DUMPFILE)|COPY|CALL|SET|LOCK|UNLOCK|"
    r"ATTACH|DETACH|PRAGMA|REINDEX|VACUUM)\b",
    re.IGNORECASE,
)

_UNSAFE_PATTERNS = [
    re.compile(r"--"),
    re.compile(r";\s*\S"),
    re.compile(r"/\*"),
    re.compile(r"\\x"),
    re.compile(r"UNION\s+SELECT", re.IGNORECASE),
]

_MAX_ROWS = int(os.getenv("TELECOM_DB_MAX_ROWS", "500"))


def _validate_sql(sql: str) -> str:
    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("SQL statement is empty")
    if re.search(r";\s*\S", cleaned):
        raise ValueError("Multi-statement queries are not allowed")
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if not re.match(r"^\s*(SELECT|WITH|EXPLAIN)\b", cleaned, re.IGNORECASE):
        raise ValueError("Only SELECT / WITH / EXPLAIN allowed")
    if _DANGEROUS_SQL.search(cleaned):
        raise ValueError("SQL contains forbidden keywords")
    for p in _UNSAFE_PATTERNS:
        if p.search(cleaned):
            raise ValueError(f"SQL contains unsafe pattern: {p.pattern}")
    if re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE) and not re.search(r"\bLIMIT\b", cleaned, re.IGNORECASE):
        cleaned += f" LIMIT {_MAX_ROWS}"
    return cleaned


def _execute_query(sql: str, db_path: str) -> list[dict[str, Any]]:
    resolved = str(BASE_DIR.parent / db_path) if not os.path.isabs(db_path) else db_path
    db_uri = f"file:{resolved}?mode=ro"
    try:
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = cur.fetchmany(_MAX_ROWS)
        return [dict(row) for row in rows]
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "只读 SQL 查询 (仅允许 SELECT/WITH)。系统自动添加 LIMIT。",
        },
    },
    "required": ["sql"],
}

server = Server("dataagent-db-server", version="3.0.0")


@server.list_tools()
async def list_tools() -> list[Tool]:
    tools: list[Tool] = []
    for cfg in _DB_CONFIGS:
        tools.append(Tool(
            name=cfg["tool_name"],
            description=cfg["description"],
            inputSchema=TOOL_SCHEMA,
        ))
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name not in _TOOL_DB_MAP:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    db_path = _TOOL_DB_MAP[name]
    sql = arguments.get("sql", "").strip()

    try:
        safe_sql = _validate_sql(sql)
        rows = _execute_query(safe_sql, db_path)
    except ValueError as e:
        return [TextContent(type="text", text=json.dumps(
            {"error": str(e), "query": sql}, ensure_ascii=False, indent=2
        ))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"Database query failed: {e}", "query": sql}, ensure_ascii=False, indent=2
        ))]

    return [TextContent(type="text", text=json.dumps(
        {"status": "ok", "rows": len(rows), "data": rows}, ensure_ascii=False, indent=2
    ))]


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

async def main():
    print(f"[MCP Server] Loaded {len(_DB_CONFIGS)} database tool(s): {list(_TOOL_DB_MAP.keys())}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
