import React, { useState } from 'react';
import { AlertCircle, Check, ChevronDown, Loader2 } from 'lucide-react';

const formatDuration = (durationMs) => {
  const totalSeconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0
    ? `${minutes}:${String(seconds).padStart(2, '0')}`
    : `${seconds}.${Math.floor((durationMs % 1000) / 100)}s`;
};

const ActivitySteps = ({ activity }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!activity?.steps?.length) return null;

  const currentStep = activity.active ? activity.steps[activity.steps.length - 1] : null;
  const summary = activity.active
    ? `${currentStep.label} (${formatDuration(currentStep.durationMs)})`
    : activity.failed
      ? 'Không thể hoàn tất các bước xử lý.'
      : `${activity.steps.length} bước đã hoàn tất (${formatDuration(activity.elapsedMs)})`;

  return (
    <div className="ml-12 mb-4 max-w-[85%] overflow-hidden rounded-xl border border-gray-200 bg-white text-sm shadow-sm">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-gray-600 transition hover:bg-gray-50"
        onClick={() => setIsExpanded((expanded) => !expanded)}
        aria-expanded={isExpanded}
      >
        {activity.active ? (
          <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-600" />
        ) : (
          <Check className={`h-4 w-4 shrink-0 ${activity.failed ? 'text-red-500' : 'text-emerald-600'}`} />
        )}
        <span className="min-w-0 flex-1 truncate">{summary}</span>
        <span className="shrink-0 text-xs text-gray-400">
          {activity.active ? 'Đang xử lý' : 'Đã hoàn tất'}
        </span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>

      {isExpanded && (
        <div className="border-t border-gray-100">
          {activity.steps.map((step) => (
            <div key={step.id} className="flex items-center gap-2 border-b border-gray-100 px-4 py-2.5 text-gray-600 last:border-b-0">
              {step.status === 'running' ? (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-600" />
              ) : step.status === 'failed' ? (
                <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
              ) : (
                <Check className="h-4 w-4 shrink-0 text-emerald-600" />
              )}
              <span className="min-w-0 flex-1">{step.label}</span>
              <span className="shrink-0 text-xs text-gray-400">
                {formatDuration(step.durationMs)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ActivitySteps;
