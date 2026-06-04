"""电信气象传播数据库初始化脚本

从 UrbanMeteorologicalData.csv 读取 400+ 城市的气象数据,
字段规范化映射后写入 SQLite 数据库 telecom_propagation.db。

用法:
    cd backend
    uv run python -m src.init_telecom_db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "UrbanMeteorologicalData.csv"
DB_PATH = BASE_DIR / "data" / "telecom_propagation.db"

# ---------------------------------------------------------------------------
# 字段映射: CSV 原始列名 → 标准化列名
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "CityCode": "city_code",
    "PinYin": "city_pinyin",
    "Name": "city_name",
    "spring_pres": "spring_pressure",
    "summer_pres": "summer_pressure",
    "autumn_pres": "autumn_pressure",
    "winter_pres": "winter_pressure",
    "spring": "spring_temp",
    "summer": "summer_temp",
    "autumn": "autumn_temp",
    "winter": "winter_temp",
    "waterVaporDensity": "city_type",  # CSV 中该列实际存储城市类型枚举 (0/1/2)
}

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS city_climate_stats (
    city_code       TEXT PRIMARY KEY,
    city_pinyin     TEXT,
    city_name       TEXT    NOT NULL,
    spring_pressure REAL,       -- 春季气压 (hPa)
    summer_pressure REAL,       -- 夏季气压 (hPa)
    autumn_pressure REAL,       -- 秋季气压 (hPa)
    winter_pressure REAL,       -- 冬季气压 (hPa)
    spring_temp     REAL,       -- 春季气温 (℃)
    summer_temp     REAL,       -- 夏季气温 (℃)
    autumn_temp     REAL,       -- 秋季气温 (℃)
    winter_temp     REAL,       -- 冬季气温 (℃)
    city_type       INTEGER     -- 城市类型: 0=内陆, 1=沿海, 2=海岛
);
"""

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV file not found: {CSV_PATH}")
        return

    # 1. 读取 CSV (ISO-8859-1 编码), 显式指定列名
    df = pd.read_csv(
        CSV_PATH,
        encoding="ISO-8859-1",
        dtype={
            "CityCode": str,
            "PinYin": str,
            "Name": str,
            "waterVaporDensity": int,
        },
    )
    print(f"[READ] CSV: {len(df)} rows, {len(df.columns)} cols")
    print(f"       Raw columns: {list(df.columns)}")

    # 2. 字段重命名
    df = df.rename(columns=COLUMN_MAP)
    expected_columns = list(COLUMN_MAP.values())
    df = df[[c for c in expected_columns if c in df.columns]]
    print(f"       Normalized: {list(df.columns)}")

    # 3. 类型修正
    df["city_code"] = df["city_code"].astype(str).str.strip()
    df["city_pinyin"] = df["city_pinyin"].astype(str).str.strip()
    df["city_name"] = df["city_name"].astype(str).str.strip()
    df["city_type"] = df["city_type"].astype(int)

    # 4. 创建 SQLite 数据库 + 表
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()

    # 5. 清空旧数据并全量灌入
    conn.execute("DELETE FROM city_climate_stats")
    df.to_sql("city_climate_stats", conn, if_exists="append", index=False)
    conn.commit()

    # 6. 验证
    count = conn.execute("SELECT COUNT(*) FROM city_climate_stats").fetchone()[0]
    type_dist = conn.execute(
        "SELECT city_type, COUNT(*) FROM city_climate_stats GROUP BY city_type ORDER BY city_type"
    ).fetchall()

    print(f"[DONE] Wrote {count} rows to {DB_PATH}")
    print(f"       Inland(city_type=0): {dict(type_dist).get(0, 0)} rows")
    print(f"       Coastal(city_type=1): {dict(type_dist).get(1, 0)} rows")
    print(f"       Island(city_type=2): {dict(type_dist).get(2, 0)} rows")

    # 7. 样本抽查
    sample = conn.execute(
        "SELECT city_name, city_pinyin, city_type, spring_temp, summer_temp "
        "FROM city_climate_stats LIMIT 5"
    ).fetchall()
    print("\n[Sample]:")
    for row in sample:
        print(f"  {row[0]} ({row[1]}) type={row[2]} spring={row[3]}C summer={row[4]}C")

    conn.close()


if __name__ == "__main__":
    main()
