import { BookText, Copy, FileText, Hash, X } from 'lucide-react';
import { RetrievalDocument } from '../types';

interface DocumentFragmentModalProps {
  document: RetrievalDocument | null;
  isOpen: boolean;
  onClose: () => void;
}

function copyText(value: string) {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(value);
  }
  return Promise.reject(new Error('Clipboard unavailable'));
}

export function DocumentFragmentModal({ document, isOpen, onClose }: DocumentFragmentModalProps) {
  if (!isOpen || !document) return null;

  const content = document.content || document.excerpt || '';
  const heading = document.title || document.source_file || '文档片段';

  return (
    <div className="fixed inset-0 z-[70] bg-slate-900/50 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-[760px] max-w-[96vw] max-h-[88vh] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl flex flex-col">
        <div className="px-5 py-4 border-b border-slate-100 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm text-slate-900">
              <BookText className="w-4 h-4 text-indigo-500" />
              <span className="truncate">{heading}</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
              {document.citation_id != null && (
                <span className="rounded-full border border-indigo-200 bg-indigo-50 px-2 py-0.5 text-indigo-700">[{document.citation_id}]</span>
              )}
              {document.category && <span>{document.category}</span>}
              {document.source_file && <span>{document.source_file}</span>}
              {document.section_path && <span>{document.section_path}</span>}
              {document.page != null && <span>第 {document.page} 页</span>}
              <span>score {document.score.toFixed(3)}</span>
              {document.relevance != null && <span>relevance {document.relevance.toFixed(3)}</span>}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                if (content) {
                  void copyText(content);
                }
              }}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-600 transition-colors"
            >
              <Copy className="w-3.5 h-3.5" />
              复制
            </button>
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-5 overflow-y-auto space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
              <div className="text-slate-500 flex items-center gap-1">
                <FileText className="w-3.5 h-3.5" />
                来源文件
              </div>
              <div className="mt-1 text-slate-800 break-all">{document.source_file || '—'}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
              <div className="text-slate-500 flex items-center gap-1">
                <Hash className="w-3.5 h-3.5" />
                片段定位
              </div>
              <div className="mt-1 text-slate-800">{document.section_path || (document.page != null ? `第 ${document.page} 页` : '—')}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
              <div className="text-slate-500">引用编号</div>
              <div className="mt-1 text-slate-800">{document.citation_id != null ? `[${document.citation_id}]` : '—'}</div>
            </div>
          </div>

          <div className="rounded-2xl border border-indigo-100 bg-indigo-50/50 px-4 py-4">
            <div className="text-xs text-slate-500 mb-2">原文片段</div>
            <div className="text-sm leading-7 text-slate-700 whitespace-pre-wrap break-words">
              {content || '暂无可显示内容'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
