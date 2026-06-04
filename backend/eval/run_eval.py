#!/usr/bin/env python
"""
DataAgent RAG 检索质量评测脚本

用法（在 backend 目录下执行）：
    python -m eval.run_eval                     # 默认评测
    python -m eval.run_eval --top_k 4           # 指定 top_k
    python -m eval.run_eval --rebuild            # 先重建索引再评测

输出指标：
    - Hit@K          ：top_k 结果中命中期望来源的比例
    - Keyword Recall ：top_k 结果中包含期望关键词的比例
    - MRR            ：首个命中结果排名的倒数均值
    - 每题详细结果
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 确保 backend 包可被 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_engine import (
    rebuild_knowledge_base,
    retrieve_knowledge,
    ensure_knowledge_base_loaded,
    get_knowledge_base_stats,
)

EVAL_DATASET = Path(__file__).resolve().parent / "eval_dataset.json"


def load_dataset() -> list[dict]:
    data = json.loads(EVAL_DATASET.read_text(encoding="utf-8"))
    return data["questions"]


def evaluate(questions: list[dict], top_k: int) -> dict:
    hit_count = 0
    keyword_recall_sum = 0.0
    mrr_sum = 0.0
    details: list[dict] = []

    for q in questions:
        hits = retrieve_knowledge(query=q["query"], top_k=top_k)
        retrieved_sources = [h.source for h in hits]
        retrieved_texts = " ".join(h.text for h in hits)

        # Hit@K
        expected = set(q.get("expected_sources", []))
        hit = any(src in expected for src in retrieved_sources)
        if hit:
            hit_count += 1

        # MRR
        first_rank = 0
        for rank, src in enumerate(retrieved_sources, start=1):
            if src in expected:
                first_rank = rank
                break
        mrr_sum += (1.0 / first_rank) if first_rank else 0.0

        # Keyword Recall
        expected_kw = q.get("expected_keywords", [])
        if expected_kw:
            kw_hits = sum(1 for kw in expected_kw if kw in retrieved_texts)
            kw_recall = kw_hits / len(expected_kw)
        else:
            kw_recall = 1.0
        keyword_recall_sum += kw_recall

        details.append({
            "id": q["id"],
            "query": q["query"],
            "type": q.get("type", ""),
            "hit": hit,
            "keyword_recall": round(kw_recall, 4),
            "retrieved_sources": retrieved_sources,
            "top1_score": round(hits[0].final_score, 4) if hits else 0.0,
        })

    n = len(questions)
    return {
        "total": n,
        "top_k": top_k,
        "hit_at_k": round(hit_count / n, 4) if n else 0,
        "keyword_recall": round(keyword_recall_sum / n, 4) if n else 0,
        "mrr": round(mrr_sum / n, 4) if n else 0,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="先重建知识库索引")
    parser.add_argument("--output", type=str, default="", help="结果输出到 JSON 文件")
    args = parser.parse_args()

    if args.rebuild:
        print("[Eval] 重建知识库索引...")
        result = rebuild_knowledge_base(reset=True)
        print(f"[Eval] 重建结果: doc={result.get('doc_count')}, chunk={result.get('chunk_count')}")
    else:
        ensure_knowledge_base_loaded()

    stats = get_knowledge_base_stats()
    print(f"[Eval] 知识库状态: doc={stats['doc_count']}, chunk={stats['chunk_count']}, "
          f"embed={stats['embedding_provider']}/{stats['embedding_model']}")

    questions = load_dataset()
    print(f"[Eval] 评测集加载完成: {len(questions)} 题, top_k={args.top_k}")
    print("-" * 60)

    result = evaluate(questions, args.top_k)

    for d in result["details"]:
        status = "✓" if d["hit"] else "✗"
        print(f"  {status}  Q{d['id']:02d} [{d['type']:10s}] kw_recall={d['keyword_recall']:.2f}  "
              f"top1={d['top1_score']:.3f}  | {d['query']}")

    print("-" * 60)
    print(f"  Hit@{result['top_k']}          = {result['hit_at_k']:.4f}")
    print(f"  Keyword Recall = {result['keyword_recall']:.4f}")
    print(f"  MRR            = {result['mrr']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
