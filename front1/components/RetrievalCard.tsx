import { BookOpen, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import { RetrievalDocument } from '../types';
import { DocumentFragmentModal } from './DocumentFragmentModal';

interface RetrievalCardProps {
  query: string;
  strategy?: string;
  documents: RetrievalDocument[];
}

export function RetrievalCard({ query, strategy, documents }: RetrievalCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<RetrievalDocument | null>(null);

  if (documents.length === 0) return null;

  return (
    <>
      <div className="my-2 rounded-lg border border-violet-200 bg-violet-50 px-3 py-2">
        <div
          className="flex items-center gap-2 text-sm cursor-pointer select-none"
          onClick={() => setExpanded((prev) => !prev)}
        >
          <BookOpen className="h-4 w-4 text-violet-600" />
          <span className="font-medium text-slate-700">知识库检索</span>
          <span className="ml-auto inline-flex items-center gap-1 text-xs text-slate-500">
            <span className="rounded-full bg-white/80 px-2 py-0.5">命中 {documents.length} 条</span>
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </span>
        </div>
        {query && (
          <div className="mt-1 text-xs text-slate-600">
            <span className="text-slate-400">查询：</span>
            {query}
            {strategy && (
              <span className="ml-2 text-[11px] text-slate-400">({strategy})</span>
            )}
          </div>
        )}
        {expanded && (
          <ol className="mt-2 space-y-1.5">
            {documents.map((doc, idx) => (
              <li key={doc.id} className="flex items-start gap-2 text-xs">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-violet-100 text-violet-600 flex items-center justify-center text-[11px] font-medium">
                  {doc.citation_id ?? idx + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-violet-700 truncate">{doc.title}</span>
                    <span className="rounded bg-white/70 px-1.5 py-0.5 text-[10px] text-slate-500 border border-slate-200">
                      {doc.category}
                    </span>
                    <span className="text-[11px] text-slate-400">
                      score {doc.score.toFixed(3)}
                    </span>
                    {doc.relevance != null && (
                      <span className="text-[11px] text-emerald-600">
                        relevance {doc.relevance.toFixed(3)}
                      </span>
                    )}
                  </div>
                  {(doc.source_file || doc.section_path || doc.page != null) && (
                    <div className="mt-0.5 text-[10px] text-slate-500">
                      {doc.source_file && <span>{doc.source_file}</span>}
                      {doc.section_path && <span>{doc.source_file ? ' · ' : ''}{doc.section_path}</span>}
                      {doc.page != null && <span>{doc.source_file || doc.section_path ? ' · ' : ''}第 {doc.page} 页</span>}
                    </div>
                  )}
                  {(doc.keyword_score != null || doc.vector_score != null) && (
                    <div className="mt-0.5 text-[10px] text-slate-400">
                      {doc.keyword_score != null && <span>关键词 {doc.keyword_score.toFixed(3)}</span>}
                      {doc.keyword_score != null && doc.vector_score != null && <span> · </span>}
                      {doc.vector_score != null && <span>向量 {doc.vector_score.toFixed(3)}</span>}
                    </div>
                  )}
                  {doc.excerpt && (
                    <div className="mt-1 rounded border border-violet-100 bg-white/70 px-2 py-1 text-[11px] leading-5 text-slate-600 whitespace-pre-wrap">
                      {doc.excerpt}
                    </div>
                  )}
                  <div className="mt-2">
                    <button
                      onClick={() => setSelectedDocument(doc)}
                      className="inline-flex items-center gap-1 rounded-md border border-violet-200 bg-white px-2 py-1 text-[11px] text-violet-700 hover:border-violet-300 hover:bg-violet-50 transition-colors"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                      查看原文片段
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>

      <DocumentFragmentModal
        document={selectedDocument}
        isOpen={selectedDocument != null}
        onClose={() => setSelectedDocument(null)}
      />
    </>
  );
}
