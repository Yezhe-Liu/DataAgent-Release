"""查询处理: Query Expansion + HyDE (Hypothetical Document Embeddings)

- QueryExpander:  同义词/缩写扩展
- HyDEGenerator:  生成假设答案，用假设答案的向量去检索
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 1. Query Expander
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "退款": ["退费", "退钱", "返还", "退", "退还"],
    "故障": ["宕机", "异常", "报错", "不可用", "崩溃"],
    "P0": ["一级", "最高优先级", "紧急"],
    "SLA": ["服务等级协议", "服务水平", "服务保障"],
    "账户": ["账号", "用户", "登录"],
    "API": ["接口", "应用程序接口"],
    "响应时间": ["延迟", "响应速度", "RT"],
}


class QueryExpander:
    """基于规则的同义词/缩写扩展。"""

    def __init__(self, synonyms: dict[str, str | list[str]] | None = None):
        self._synonyms = synonyms or _SYNONYMS

    def expand(self, query: str) -> list[str]:
        """生成原始查询 + 扩展变体。"""
        variants = [query]
        for term, expansions in self._synonyms.items():
            exps = expansions if isinstance(expansions, list) else [expansions]
            for exp in exps:
                if term.lower() in query.lower() and exp.lower() not in query.lower():
                    variants.append(query.replace(term, exp))
                elif exp.lower() in query.lower() and term.lower() not in query.lower():
                    variants.append(query.replace(exp, term))
        return variants


# ---------------------------------------------------------------------------
# 2. HyDE Generator
# ---------------------------------------------------------------------------

HYDE_PROMPT = """你是一个技术文档撰写助手。请根据以下问题，写一段假设的文档片段来回答它。
文档片段应模仿企业知识库文档的风格（客观、准确、包含细节）。

问题: {query}

假设文档片段:"""


class HyDEGenerator:
    """生成假设文档片段，用假设片段的向量替代查询向量做检索。"""

    def __init__(self, model: Any):
        self.model = model

    def generate(self, query: str) -> str:
        """生成 HyDE 假设文档。"""
        try:
            response = self.model.invoke(HYDE_PROMPT.format(query=query))
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return query  # fallback: 用原查询


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class QueryProcessor:
    """统一查询处理入口。"""

    def __init__(self, model: Any = None, expander: QueryExpander | None = None):
        self.expander = expander or QueryExpander()
        self.hyde = HyDEGenerator(model) if model else None

    def process(self, query: str, use_expand: bool = True, use_hyde: bool = False) -> list[str]:
        """返回用于检索的查询列表。"""
        queries = self.expander.expand(query) if use_expand else [query]

        if use_hyde and self.hyde:
            hyde_text = self.hyde.generate(query)
            queries.append(hyde_text)

        return queries
