import { CheckCircle2, Loader2, Route } from 'lucide-react';

interface WorkflowTraceCardProps {
  stage: string;
  message: string;
  status: 'running' | 'completed';
  route?: string;
}

export function WorkflowTraceCard({ stage, message, status, route }: WorkflowTraceCardProps) {
  const isRunning = status === 'running';
  const normalizedStage = stage.toLowerCase();
  const isFinalSuccess = !isRunning && (
    normalizedStage === 'finalize' ||
    (normalizedStage === 'answer' && /已生成最终回答|最终回答|最终答复|完成回答/.test(message))
  );
  const containerClass = isRunning
    ? 'border-amber-200 bg-amber-50'
    : isFinalSuccess
      ? 'border-emerald-200 bg-emerald-50'
      : 'border-sky-200 bg-sky-50';
  const routeClass = isRunning
    ? 'bg-white/80 text-amber-700'
    : isFinalSuccess
      ? 'bg-white text-emerald-700'
      : 'bg-white text-sky-700';
  const messageClass = isRunning
    ? 'text-slate-600'
    : isFinalSuccess
      ? 'text-slate-600'
      : 'text-slate-500';

  return (
    <div className={`my-2 rounded-lg border px-3 py-2 ${containerClass}`}>
      <div className="flex items-center gap-2 text-sm">
        {isRunning ? (
          <Loader2 className="h-4 w-4 animate-spin text-amber-600" />
        ) : (
          <CheckCircle2 className={`h-4 w-4 ${isFinalSuccess ? 'text-emerald-600' : 'text-sky-600'}`} />
        )}
        <span className="font-medium text-slate-700">工作流节点：{stage}</span>
        {route ? (
          <span className={`ml-auto inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs ${routeClass}`}>
            <Route className="h-3 w-3" />
            {route}
          </span>
        ) : null}
      </div>
      <div className={`mt-1 text-xs leading-5 ${messageClass}`}>
        {message}
      </div>
    </div>
  );
}
