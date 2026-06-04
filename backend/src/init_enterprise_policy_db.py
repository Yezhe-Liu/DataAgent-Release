"""企业政策数据库初始化

创建 enterprise_policy.db，包含 policies / compliance_rules / departments 三表，
演示 DataAgent 在企业制度、合规管理场景的扩展能力。

用法:
    cd backend
    uv run python -m src.init_enterprise_policy_db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "enterprise_policy.db"

DDL = """
CREATE TABLE IF NOT EXISTS departments (
    dept_id     INTEGER PRIMARY KEY,
    dept_name   TEXT NOT NULL,
    manager     TEXT,
    headcount   INTEGER
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id       INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,
    content         TEXT,
    department      TEXT,
    effective_date  TEXT,
    version         TEXT
);

CREATE TABLE IF NOT EXISTS compliance_rules (
    rule_id     INTEGER PRIMARY KEY,
    rule_name   TEXT NOT NULL,
    category    TEXT NOT NULL,
    description TEXT,
    risk_level  TEXT,
    policy_id   INTEGER REFERENCES policies(policy_id)
);
"""

DEPARTMENTS = [
    (1, "技术研发部", "张伟", 45),
    (2, "信息安全部", "李娜", 12),
    (3, "人力资源部", "王芳", 8),
    (4, "财务部", "陈刚", 10),
    (5, "法务合规部", "刘洋", 6),
    (6, "市场运营部", "赵雪", 20),
]

POLICIES = [
    (1, "数据安全管理办法", "安全",
     "为保障公司数据资产安全，规范数据分类分级管理，明确数据访问权限与审批流程...",
     "信息安全部", "2024-01-15", "v2.1"),
    (2, "员工考勤与休假制度", "人事",
     "员工每日工作时间为 09:00-18:00，午休 12:00-13:00。年假按工龄计算：1-5年 5天，5-10年 10天，10年以上 15天...",
     "人力资源部", "2024-03-01", "v3.0"),
    (3, "财务报销管理规范", "财务",
     "差旅报销需提前提交申请单，住宿标准一线城市不超过 500元/天，二线城市不超过 350元/天...",
     "财务部", "2024-02-10", "v1.5"),
    (4, "代码审查与上线流程", "技术",
     "所有生产环境代码变更必须经过至少一名 Senior 工程师 Code Review。上线窗口为每周二、四 20:00-22:00...",
     "技术研发部", "2024-04-01", "v2.0"),
    (5, "个人信息保护合规指引", "合规",
     "依据《个人信息保护法》第13条，收集用户个人信息须取得明确同意，敏感信息需单独授权...",
     "法务合规部", "2024-05-20", "v1.0"),
    (6, "网络安全应急响应预案", "安全",
     "P0 级安全事件：15分钟内拉起应急小组，30分钟内完成初步隔离，2小时内向监管部门报告...",
     "信息安全部", "2024-06-01", "v2.3"),
    (7, "供应商准入与评估制度", "行政",
     "新进供应商需通过资质审查、样品测试、现场考察三阶段评估，综合评分 80 分以上方可入库...",
     "法务合规部", "2024-01-10", "v1.2"),
    (8, "远程办公管理办法", "人事",
     "远程办公需提前一天申请，核心研发岗每周不超过 2 天远程，需确保 VPN 接入和代码安全...",
     "人力资源部", "2024-07-01", "v1.0"),
]

COMPLIANCE_RULES = [
    (1, "数据最小化原则", "数据隐私",
     "收集个人信息应当限于实现处理目的的最小范围，不得过度收集", "高", 5),
    (2, "跨境数据传输审批", "数据隐私",
     "向境外提供个人信息须通过国家网信部门安全评估", "严重", 5),
    (3, "安全事件分级响应", "安全审计",
     "P0(严重): 影响>1000用户或涉及敏感信息泄露; P1(高): 100-1000用户; P2(中): <100用户", "严重", 6),
    (4, "代码变更审计留痕", "安全审计",
     "所有代码提交必须关联 TAPD 工单，变更记录保存至少 3 年", "中", 4),
    (5, "财务凭证归档期限", "财务合规",
     "原始凭证保存期限不少于 30 年，电子凭证需多副本异地备份", "中", 3),
    (6, "加班工时上限", "劳动法规",
     "每月加班不超过 36 小时，每日不超过 3 小时，需员工自愿签署加班确认书", "高", 2),
    (7, "第三方数据共享协议", "数据隐私",
     "向第三方共享用户数据前须签订数据处理协议(DPA)，明确数据使用范围与删除期限", "严重", 5),
    (8, "反商业贿赂条款", "财务合规",
     "员工不得接受供应商价值超过 200 元的礼品，商务宴请人均不超过 300 元", "高", 7),
]


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(DDL)

    conn.execute("DELETE FROM compliance_rules")
    conn.execute("DELETE FROM policies")
    conn.execute("DELETE FROM departments")

    conn.executemany(
        "INSERT INTO departments VALUES (?, ?, ?, ?)", DEPARTMENTS
    )
    conn.executemany(
        "INSERT INTO policies VALUES (?, ?, ?, ?, ?, ?, ?)", POLICIES
    )
    conn.executemany(
        "INSERT INTO compliance_rules VALUES (?, ?, ?, ?, ?, ?)", COMPLIANCE_RULES
    )
    conn.commit()

    dept_n = conn.execute("SELECT COUNT(*) FROM departments").fetchone()[0]
    pol_n = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
    rule_n = conn.execute("SELECT COUNT(*) FROM compliance_rules").fetchone()[0]

    print(f"[DONE] enterprise_policy.db created at {DB_PATH}")
    print(f"       departments: {dept_n} rows")
    print(f"       policies: {pol_n} rows")
    print(f"       compliance_rules: {rule_n} rows")

    # 样本
    for row in conn.execute(
        "SELECT title, category FROM policies ORDER BY policy_id LIMIT 3"
    ).fetchall():
        print(f"       - [{row[1]}] {row[0]}")

    conn.close()


if __name__ == "__main__":
    main()
