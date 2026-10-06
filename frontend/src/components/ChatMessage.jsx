import React from 'react';
import ShapChart from './ShapChart';
import { Bot, User } from 'lucide-react';

const ChatMessage = ({ message }) => {
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
            <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
          </div>
          
          {/* Render SHAP Charts if available */}
          {message.forecast_results && (
            <div className="flex flex-col gap-4 w-[400px] max-w-full">
              <ShapChart data={message.forecast_results.revenue} title="Động lực Doanh Thu (VND)" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
