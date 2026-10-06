import React, { useState } from 'react';
import { useChat } from '../hooks/useChat';
import ChatMessage from '../components/ChatMessage';
import { Send, Loader2, BarChart3 } from 'lucide-react';

const AgentChatPage = () => {
  const { messages, isLoading, sendMessage, messagesEndRef } = useChat();
  const [inputText, setInputText] = useState('');

  const handleSend = (e) => {
    e.preventDefault();
    sendMessage(inputText);
    setInputText('');
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      
      {/* Sidebar (Optional) */}
      <div className="w-64 bg-slate-900 text-white p-4 hidden md:flex flex-col">
        <div className="flex items-center gap-2 mb-8 font-bold text-lg text-blue-400">
          <BarChart3 /> Forecast Agent
        </div>
        <p className="text-sm text-slate-400">Hệ thống phân tích & dự báo doanh thu ứng dụng XAI (SHAP).</p>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full max-w-4xl mx-auto border-x border-gray-200 bg-gray-50/50 shadow-xl">
        
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
          <h1 className="font-semibold text-gray-800">Trợ lý Dự báo (AI)</h1>
          <span className="flex items-center gap-2 text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Online
          </span>
        </header>

        {/* Message List */}
        <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          
          {isLoading && (
            <div className="flex gap-2 items-center text-gray-400 text-sm ml-12">
              <Loader2 className="animate-spin w-4 h-4" /> 
              <span>AI đang phân tích dữ liệu...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-white border-t border-gray-200">
          <form onSubmit={handleSend} className="relative flex items-center">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Nhập câu hỏi dự báo..."
              className="w-full bg-gray-100 border-none text-sm text-gray-800 rounded-full pl-6 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!inputText.trim() || isLoading}
              className="absolute right-2 p-2.5 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={18} />
            </button>
          </form>
        </div>

      </div>
    </div>
  );
};

export default AgentChatPage;
