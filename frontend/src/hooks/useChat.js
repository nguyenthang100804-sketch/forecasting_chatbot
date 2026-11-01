import { useRef, useEffect, useState } from 'react';
import { useCopilotChatInternal } from '@copilotkit/react-core';

const FORECAST_DATA_PATTERN = /<!--FORECAST_DATA:(.*?)-->/s;

const getMessageContent = (message) => {
  const content = message?.content;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';

  return content
    .filter((part) => part?.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text)
    .join('\n');
};

const removeForecastMarker = (content) =>
  content.replace(FORECAST_DATA_PATTERN, '').trim();

const PHASES = {
  thinking: 'LLM đang phân tích yêu cầu và chọn hành động.',
  tool_calling: 'LLM đang gọi tool xử lý dữ liệu.',
  tool_running: 'Backend đang chạy tool dự báo/giải thích.',
  composing: 'LLM đang tổng hợp câu trả lời cuối cùng.',
};

const getToolCalls = (message) => {
  if (Array.isArray(message?.tool_calls)) return message.tool_calls;
  if (Array.isArray(message?.toolCalls)) return message.toolCalls;
  if (!Array.isArray(message?.content)) return [];

  return message.content.filter((part) =>
    part?.type === 'tool-call' || part?.type === 'tool_call',
  );
};

const getToolName = (message) => {
  const toolCall = getToolCalls(message)[0];
  return toolCall?.name || toolCall?.function?.name || message?.name || '';
};

const getBackendPhase = (messages, isLoading) => {
  const latestMessage = messages[messages.length - 1];
  if (latestMessage?.role === 'tool' || latestMessage?.type === 'tool') {
    const toolName = getToolName(latestMessage);
    const workflow = toolName === 'explain_forecast'
      ? 'feature_node → SHAP'
      : 'feature_node → ml_node';
    return {
      id: 'tool_running',
      label: `Backend đang chạy ${toolName || 'tool dự báo'} (${workflow}).`,
    };
  }
  const toolName = getToolName(latestMessage);
  if (getToolCalls(latestMessage).length > 0) {
    return {
      id: 'tool_calling',
      label: `LLM đang gọi tool ${toolName || 'xử lý dữ liệu'}.`,
    };
  }
  if (isLoading && latestMessage?.role === 'assistant') {
    return { id: 'composing', label: PHASES.composing };
  }
  return { id: 'thinking', label: PHASES.thinking };
};

const createActivity = (content) => {
  const isExplanation = /tại sao|giải thích|chi tiết|vì sao|phân tích/i.test(content);
  const startedAt = Date.now();

  return {
    active: true,
    startedAt,
    elapsedMs: 0,
    isExplanation,
    currentPhase: {
      id: 'thinking',
      label: PHASES.thinking,
      startedAt,
    },
    steps: [{
      id: 'thinking',
      label: PHASES.thinking,
      status: 'running',
      startedAt,
      durationMs: 0,
    }],
  };
};

export const useChat = () => {
  const {
    messages: copilotMessages,
    sendMessage: sendCopilotMessage,
    isLoading,
  } = useCopilotChatInternal();
  const messagesEndRef = useRef(null);
  const [error, setError] = useState('');
  const [activity, setActivity] = useState(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [copilotMessages]);

  const sendMessage = async (text) => {
    const content = text.trim();
    if (!content) return;

    setError('');
    setActivity(createActivity(content));

    try {
      await sendCopilotMessage({
        id: crypto.randomUUID(),
        role: 'user',
        content,
      });
      setActivity((currentActivity) =>
        currentActivity
          ? {
              ...currentActivity,
              active: false,
              elapsedMs: Date.now() - currentActivity.startedAt,
              completed: true,
              steps: currentActivity.steps.map((step, index) => ({
                ...step,
                status: 'completed',
                durationMs: index === currentActivity.steps.length - 1
                  ? Date.now() - step.startedAt
                  : step.durationMs,
                endedAt: index === currentActivity.steps.length - 1
                  ? Date.now()
                  : step.endedAt,
              })),
            }
          : currentActivity,
      );
    } catch (sendError) {
      console.error('Không thể gửi tin nhắn tới Forecast Agent:', sendError);
      setError('Không thể kết nối tới Forecast Agent. Vui lòng kiểm tra backend và thử lại.');
      setActivity((currentActivity) => {
        if (!currentActivity) return currentActivity;
        const now = Date.now();
        return {
          ...currentActivity,
          active: false,
          failed: true,
          elapsedMs: now - currentActivity.startedAt,
          steps: currentActivity.steps.map((step, index) =>
            index === currentActivity.steps.length - 1
              ? {
                  ...step,
                  status: 'failed',
                  endedAt: now,
                  durationMs: now - step.startedAt,
                }
              : step,
          ),
        };
      });
      throw sendError;
    }
  };

  useEffect(() => {
    if (!activity?.active || !isLoading) return undefined;

    const timer = window.setInterval(() => {
      setActivity((currentActivity) => {
        if (!currentActivity?.active) return currentActivity;

        return {
          ...currentActivity,
          elapsedMs: Date.now() - currentActivity.startedAt,
          steps: currentActivity.steps.map((step, index) =>
            index === currentActivity.steps.length - 1 && step.status === 'running'
              ? { ...step, durationMs: Date.now() - step.startedAt }
              : step,
          ),
        };
      });
    }, 250);

    return () => window.clearInterval(timer);
  }, [activity?.active, isLoading]);

  useEffect(() => {
    if (!activity?.active || !isLoading) return;

    const nextPhase = getBackendPhase(
      Array.isArray(copilotMessages) ? copilotMessages : [],
      isLoading,
    );
    if (nextPhase.id === activity.currentPhase.id) return;

    const now = Date.now();
    setActivity((currentActivity) => {
      if (!currentActivity?.active || currentActivity.currentPhase.id === nextPhase.id) {
        return currentActivity;
      }

      const steps = currentActivity.steps.map((step, index) =>
        index === currentActivity.steps.length - 1
          ? {
              ...step,
              status: 'completed',
              endedAt: now,
              durationMs: now - step.startedAt,
            }
          : step,
      );
      return {
        ...currentActivity,
        currentPhase: {
          id: nextPhase.id,
          label: nextPhase.label,
          startedAt: now,
        },
        steps: [
          ...steps,
          {
            id: `${nextPhase.id}-${now}`,
            label: nextPhase.label,
            status: 'running',
            startedAt: now,
            durationMs: 0,
          },
        ],
      };
    });
  }, [activity?.active, activity?.currentPhase?.id, copilotMessages, isLoading]);

  // Ánh xạ tin nhắn của CopilotKit về format UI cũ
  const safeMessages = Array.isArray(copilotMessages) ? copilotMessages : [];
  const forecastData = (() => {
    const lastUserIndex = safeMessages.reduce(
      (lastIndex, message, index) => (message?.role === 'user' ? index : lastIndex),
      -1,
    );
    const lastAssistantIndex = safeMessages.reduce(
      (lastIndex, message, index) => (message?.role === 'assistant' ? index : lastIndex),
      -1,
    );

    if (lastAssistantIndex <= lastUserIndex) return null;

    const assistantMessage = safeMessages[lastAssistantIndex];
    const match = getMessageContent(assistantMessage).match(FORECAST_DATA_PATTERN);
    if (!match) return null;

    try {
      const payload = JSON.parse(match[1]);
      return payload.forecast_results || payload;
    } catch (parseError) {
      console.error('Không thể đọc dữ liệu biểu đồ dự báo:', parseError);
      return null;
    }
  })();
  const latestForecastMessage = (() => {
    for (let index = safeMessages.length - 1; index >= 0; index -= 1) {
      if (safeMessages[index]?.role === 'assistant' &&
          FORECAST_DATA_PATTERN.test(getMessageContent(safeMessages[index]))) {
        return safeMessages[index];
      }
    }
    return null;
  })();
  const visibleMessages = safeMessages.filter(
    (msg) =>
      (msg?.role === 'user' || msg?.role === 'assistant') &&
      removeForecastMarker(getMessageContent(msg)).length > 0,
  );
  const lastBotIndex = visibleMessages.reduce(
    (lastIndex, msg, index) => (msg?.role === 'assistant' ? index : lastIndex),
    -1,
  );
  const mappedMessages = visibleMessages.map((msg, index) => {
    const previousUserMessage = [...visibleMessages]
      .slice(0, index)
      .reverse()
      .find((previousMessage) => previousMessage.role === 'user');
    const isExplanation = previousUserMessage
      ? /tại sao|giải thích|chi tiết|vì sao|phân tích/i.test(
          getMessageContent(previousUserMessage),
        )
      : false;

    return {
      id: msg.id || index.toString(),
      role: msg.role === 'user' ? 'user' : 'bot',
      text: removeForecastMarker(getMessageContent(msg)),
      forecast_results: index === lastBotIndex && isExplanation ? forecastData : null,
      hasForecast: msg === latestForecastMessage,
      isExplanation,
    };
  });

  // Mặc định luôn có 1 câu chào (giống UI cũ)
  if (mappedMessages.length === 0) {
      mappedMessages.push({
          id: 'welcome',
          role: 'bot',
          text: 'Xin chào! Mình là trợ lý dự báo doanh thu. Bạn muốn dự báo cho thời điểm nào sắp tới?',
          isWelcome: true,
      });
  }

  return {
    messages: mappedMessages,
    isLoading: !!isLoading,
    activity,
    error,
    sendMessage,
    messagesEndRef,
  };
};
