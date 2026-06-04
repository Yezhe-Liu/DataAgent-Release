import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import { useState } from 'react';

interface StepCardProps {
  stepId: string;
  tool: string;
  goal: string;
  args?: any;
  output?: string;
  error?: string;
  durationMs?: number;
  status: 'running' | 'completed' | 'failed';
}

export function StepCard({ stepId, tool, goal, args, output, error, durationMs, status }: StepCardProps) {
  const [expanded, setExpanded] = useState(false);

  const statusConfig = {
    running: { icon: Loader2, color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-200', label: '执行中' },
    completed: { icon: CheckCircle2, color: 'text-emerald-600', bg: 'bg-emerald-50', border: 'border-emerald-200', label: '完成' },
    failed: { icon: XCircle, color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200', label: '失败' },
  };

  const cfg = statusConfig[status];
  const Icon = cfg.icon;

  return (
    <div className={`my-1.5 rounded-lg border px-3 py-2 ${cfg.border} ${cfg.bg}`}>
      <div
        className="flex items-center gap-2 text-sm cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        {status === 'running' ? (
          <Loader2 className={`h-4 w-4 animate-spin ${cfg.color}`} />
        ) : (
          <Icon className={`h-4 w-4 ${cfg.color}`} />
        )}
        <span className="font-medium text-slate-700">[{stepId}] {tool}</span>
        {durationMs != null && (
          <span className="text-xs text-slate-400">{durationMs}ms</span>
        )}
        <span className={`ml-auto text-xs ${cfg.color}`}>{cfg.label}</span>
      </div>
      {goal && (
        <div className="mt-0.5 text-xs text-slate-500 pl-6">{goal}</div>
      )}
      {expanded && (
        <div className="mt-2 pl-6 space-y-1 text-xs">
          {args && (
            <div>
              <span className="font-medium text-slate-500">参数：</span>
              <pre className="mt-0.5 p-2 rounded bg-white/70 text-slate-600 overflow-x-auto max-h-32 text-[11px]">
                {typeof args === 'string' ? args : JSON.stringify(args, null, 2)}
              </pre>
            </div>
          )}
          {output && (
            <div>
              <span className="font-medium text-slate-500">输出：</span>
              <pre className="mt-0.5 p-2 rounded bg-white/70 text-slate-600 overflow-x-auto max-h-40 text-[11px]">
                {output}
              </pre>
            </div>
          )}
          {error && (
            <div>
              <span className="font-medium text-red-500">错误：</span>
              <span className="text-red-600">{error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
