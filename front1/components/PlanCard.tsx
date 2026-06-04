import { ListChecks } from 'lucide-react';

interface PlanCardProps {
  rationale: string;
  steps: { id: string; tool: string; goal: string }[];
}

export function PlanCard({ rationale, steps }: PlanCardProps) {
  if (steps.length === 0) return null;

  return (
    <div className="my-2 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2">
      <div className="flex items-center gap-2 text-sm">
        <ListChecks className="h-4 w-4 text-indigo-600" />
        <span className="font-medium text-slate-700">执行计划</span>
        <span className="ml-auto text-xs text-slate-500">{steps.length} 步</span>
      </div>
      {rationale && (
        <div className="mt-1 text-xs leading-5 text-slate-600">{rationale}</div>
      )}
      <ol className="mt-1.5 space-y-1">
        {steps.map((step, idx) => (
          <li key={step.id} className="flex items-start gap-2 text-xs text-slate-600">
            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center text-[11px] font-medium mt-0.5">
              {idx + 1}
            </span>
            <div>
              <span className="font-medium text-indigo-700">{step.tool}</span>
              {step.goal && <span className="ml-1 text-slate-500">— {step.goal}</span>}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
