import { Brain, MessageSquare, BookOpen, Wrench } from 'lucide-react';

interface IntentCardProps {
  mode: 'simple_qa' | 'retrieval_qa' | 'tool_task';
  reason: string;
  candidateTools: string[];
}

const MODE_CONFIG = {
  simple_qa: { label: '直接回答', icon: MessageSquare, color: 'text-sky-600', bg: 'bg-sky-50', border: 'border-sky-200' },
  retrieval_qa: { label: '知识库检索', icon: BookOpen, color: 'text-violet-600', bg: 'bg-violet-50', border: 'border-violet-200' },
  tool_task: { label: '工具调用', icon: Wrench, color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-200' },
};

export function IntentCard({ mode, reason, candidateTools }: IntentCardProps) {
  const cfg = MODE_CONFIG[mode] || MODE_CONFIG.simple_qa;
  const Icon = cfg.icon;

  return (
    <div className={`my-2 rounded-lg border px-3 py-2 ${cfg.border} ${cfg.bg}`}>
      <div className="flex items-center gap-2 text-sm">
        <Brain className={`h-4 w-4 ${cfg.color}`} />
        <span className="font-medium text-slate-700">意图分析</span>
        <span className={`ml-auto inline-flex items-center gap-1 rounded-full bg-white/80 px-2 py-0.5 text-xs ${cfg.color}`}>
          <Icon className="h-3 w-3" />
          {cfg.label}
        </span>
      </div>
      {reason && (
        <div className="mt-1 text-xs leading-5 text-slate-600">{reason}</div>
      )}
      {candidateTools.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {candidateTools.map((tool) => (
            <span key={tool} className="inline-block rounded bg-white/70 px-1.5 py-0.5 text-[11px] text-slate-500 border border-slate-200">
              {tool}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
