import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

// Tạo 1 session ID ngẫu nhiên không cần thư viện uuid để tối giản
const generateSessionId = () => Math.random().toString(36).substring(2, 15);

export const useChat = () => {
  const [sessionId] = useState(generateSessionId());
  const [messages, setMessages] = useState([
    { id: '1', role: 'bot', text: 'Xin chào! Mình là trợ lý dự báo doanh thu. Bạn muốn dự báo cho thời điểm nào sắp tới?' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (text) => {
    if (!text.trim()) return;

    const userMsg = { id: Date.now().toString(), role: 'user', text };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      // Dùng biến môi trường trên Vercel, hoặc fallback về localhost khi dev
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.post(`${API_URL}/api/chat`, {
        session_id: sessionId,
        message: text
      });

      const { data } = response;
      
      const botMsg = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        text: data.message,
        forecast_results: data.forecast_results || null
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Lỗi API:", error);
      const errorMsg = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        text: "Xin lỗi, đã có lỗi kết nối tới máy chủ dự báo."
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return { messages, isLoading, sendMessage, messagesEndRef };
};
