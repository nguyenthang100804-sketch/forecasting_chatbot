import React, { useState } from 'react';
import { Check, ChevronDown, Loader2 } from 'lucide-react';

const ActivitySteps = ({ activity }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!activity?.steps?.length) return null;

  const visibleIndex = activity.active
    ? activity.currentIndex
    : activity.steps.length - 1;
  const completedSteps = activity.active
    ? activity.steps.slice(0, visibleIndex)
    : activity.steps;
  const currentStep = activity.active ? activity.steps[visibleIndex] : null;
  const summary = activity.active
    ? currentStep
    : activity.failed
      ? 'Không thể hoàn tất các bước xử lý.'
      : `${activity.steps.length} bước đã hoàn tất`;

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
          {completedSteps.map((step) => (
            <div key={step} className="flex items-center gap-2 border-b border-gray-100 px-4 py-2.5 text-gray-600 last:border-b-0">
              <Check className="h-4 w-4 shrink-0 text-emerald-600" />
              <span>{step}</span>
            </div>
          ))}
          {currentStep && (
            <div className="flex items-center gap-2 border-b border-gray-100 px-4 py-2.5 text-gray-700 last:border-b-0">
              <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-600" />
              <span>{currentStep}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ActivitySteps;
