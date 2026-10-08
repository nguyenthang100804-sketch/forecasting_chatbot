import React from 'react';
import ReactMarkdown from 'react-markdown';
import ShapChart from './ShapChart';
import { Bot, User } from 'lucide-react';

const ChatMessage = ({ message, onQuickAction, isLoading }) => {
  const isBot = message.role === 'bot';

  return (
    <div className={`flex w-full ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
      <div className={`flex max-w-[85%] ${isBot ? 'flex-row' : 'flex-row-reverse'} gap-3`}>
        
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${isBot ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}>
          {isBot ? <Bot size={18} /> : <User size={18} />}
        </div>

        {/* Bubble */}
        <div className="flex flex-col gap-2">
          <div className={`px-4 py-3 rounded-2xl text-sm ${isBot ? 'bg-white border border-gray-200 text-gray-800 rounded-tl-none shadow-sm' : 'bg-blue-600 text-white rounded-tr-none shadow-md'}`}>
            <div className="leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0 [&_ul]:list-disc [&_ul]:pl-5">
              <ReactMarkdown>{message.text}</ReactMarkdown>
            </div>
          </div>
          
          {/* Render SHAP Charts if available */}
          {message.forecast_results && (
            <div className="flex flex-col gap-4 w-[400px] max-w-full">
              <ShapChart data={message.forecast_results.revenue} title="Động lực Doanh Thu (VND)" />
            </div>
          )}

          {message.isWelcome && (
            <button
              type="button"
              className="w-fit rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => onQuickAction('start-prediction')}
              disabled={isLoading}
            >
              Hãy dự đoán doanh thu
            </button>
          )}

          {message.hasForecast && (
            <div className="flex w-[400px] max-w-full flex-col gap-2">
              <button
                type="button"
                className="w-full rounded-lg border border-blue-600 bg-white px-4 py-2 text-left text-sm font-medium text-blue-700 transition hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={() => onQuickAction('new-prediction')}
                disabled={isLoading}
              >
                Dự đoán ngày khác
              </button>
              {!message.isExplanation && (
                <button
                  type="button"
                  className="w-full rounded-lg border border-blue-600 bg-white px-4 py-2 text-left text-sm font-medium text-blue-700 transition hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={() => onQuickAction('explain')}
                  disabled={isLoading}
                >
                  Giải thích kết quả dự đoán
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
