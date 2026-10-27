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

const createActivity = (content) => {
  const isExplanation = /tại sao|giải thích|chi tiết|vì sao|phân tích/i.test(content);

  return {
    active: true,
    steps: isExplanation
      ? [
          'Đang phân tích yêu cầu giải thích.',
          'Đang chuẩn bị dữ liệu dự báo gần nhất.',
          'Đang tính toán các yếu tố ảnh hưởng bằng SHAP.',
          'Đang tổng hợp phần giải thích kết quả.',
        ]
      : [
          'Đang phân tích yêu cầu dự báo.',
          'Đang chuẩn bị dữ liệu và đặc trưng thời gian.',
          'Đang chạy mô hình dự báo doanh thu và chi phí.',
          'Đang tổng hợp kết quả dự báo.',
        ],
    currentIndex: 0,
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
              currentIndex: currentActivity.steps.length - 1,
              completed: true,
            }
          : currentActivity,
      );
    } catch (sendError) {
      console.error('Không thể gửi tin nhắn tới Forecast Agent:', sendError);
      setError('Không thể kết nối tới Forecast Agent. Vui lòng kiểm tra backend và thử lại.');
      setActivity((currentActivity) =>
        currentActivity ? { ...currentActivity, active: false, failed: true } : currentActivity,
      );
      throw sendError;
    }
  };

  useEffect(() => {
    if (!activity?.active) return undefined;

    const timer = window.setInterval(() => {
      setActivity((currentActivity) => {
        if (!currentActivity?.active) return currentActivity;

        return {
          ...currentActivity,
          currentIndex: Math.min(
            currentActivity.currentIndex + 1,
            currentActivity.steps.length - 1,
          ),
        };
      });
    }, 1800);

    return () => window.clearInterval(timer);
  }, [activity?.active]);

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
