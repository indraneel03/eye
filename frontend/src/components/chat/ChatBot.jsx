import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User } from 'lucide-react';

const Chatbot = ({ apiKey, modelType = 'openai' }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Scroll to the bottom of the chat when new messages are added
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let apiUrl = '';
      let requestBody = {};
      let headers = { 'Content-Type': 'application/json' };

      if (modelType === 'openai') {
        apiUrl = 'https://api.openai.com/v1/chat/completions';
        headers['Authorization'] = `Bearer ${apiKey}`;
        requestBody = {
          model: 'gpt-4', // You can change this to 'gpt-3.5-turbo' if needed
          messages: [{ role: 'user', content: input }],
          temperature: 0.7,
        };
      } else if (modelType === 'gemini') {
        apiUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=' + apiKey;
        requestBody = { contents: [{ parts: [{ text: input }] }] };
      }

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      let botReply = 'Sorry, no response.';

      if (modelType === 'openai') {
        botReply = data.choices?.[0]?.message?.content || botReply;
      } else if (modelType === 'gemini') {
        botReply = data.candidates?.[0]?.content?.parts?.[0]?.text || botReply;
      }

      const botMessage = { text: botReply, sender: 'bot' };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        { text: 'Failed to get a response. Please try again.', sender: 'bot' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-4 right-4 w-96 bg-white/10 backdrop-blur-sm rounded-lg shadow-lg border border-gray-700">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-blue-400 flex items-center">
            <Bot className="mr-2" /> Retinal Disease Chatbot
          </h3>
        </div>
        <div className="h-64 overflow-y-auto mb-4">
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              className={`flex ${
                msg.sender === 'user' ? 'justify-end' : 'justify-start'
              } mb-2`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div
                className={`flex items-start space-x-2 max-w-[80%] ${
                  msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'
                }`}
              >
                <div
                  className={`p-3 rounded-lg ${
                    msg.sender === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-100'
                  }`}
                >
                  <p className="text-sm">{msg.text}</p>
                </div>
                <div
                  className={`w-8 h-8 flex items-center justify-center rounded-full ${
                    msg.sender === 'user' ? 'bg-blue-600' : 'bg-gray-700'
                  }`}
                >
                  {msg.sender === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
              </div>
            </motion.div>
          ))}
          {isLoading && (
            <motion.div
              className="flex justify-start"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-700">
                <Bot className="w-4 h-4 text-white animate-pulse" />
              </div>
              <div className="p-3 rounded-lg bg-gray-700 text-gray-100 ml-2">
                <p className="text-sm">Typing...</p>
              </div>
            </motion.div>
          )}
          <div ref={chatEndRef} />
        </div>
        <div className="flex">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            className="flex-1 p-2 rounded-l-lg bg-gray-700/20 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-600"
            placeholder="Ask about retinal diseases..."
          />
          <button
            onClick={handleSendMessage}
            className="p-2 bg-blue-600 rounded-r-lg hover:bg-blue-700 transition-all duration-300"
          >
            <Send className="h-5 w-5 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;