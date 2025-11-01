import React, { useState, useEffect, useRef } from 'react';
import * as genAI from '@google/genai';
import type { County, ChatMessage } from '../types';

interface ChatProps {
  selectedCounty: County | null;
  onError: (message: string) => void;
}

const Chat: React.FC<ChatProps> = ({ selectedCounty, onError }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatSessionRef = useRef<genAI.Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize the Gemini chat session
  useEffect(() => {
    if (!process.env.API_KEY) {
      const errorMsg = "API key is not configured. Please ensure the API_KEY environment variable is set.";
      console.error(errorMsg);
      setError(errorMsg);
      onError(`Chat Init: ${errorMsg}`);
      return;
    }

    try {
      const ai = new genAI.GoogleGenAI({ apiKey: process.env.API_KEY as string });
      chatSessionRef.current = ai.chats.create({
        model: 'gemini-2.5-flash',
        config: {
          systemInstruction: 'You are an expert geographer and data analyst. You provide clear, concise, and interesting information about US counties based on their FIPS code. Do not mention the FIPS code unless asked. Start by giving the county name and state.',
        },
      });
      setMessages([{ author: 'system', content: "Select a county on the map to begin." }]);
      setError(null);
    } catch (e) {
      const errorMsg = "Could not initialize AI chat. The provided API key might be invalid.";
      console.error("Failed to initialize Gemini AI:", e);
      setError(errorMsg);
      onError(`Chat Init: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [onError]);

  // Effect to handle new county selection
  useEffect(() => {
    if (selectedCounty?.id) {
      // Don't proceed if chat is not initialized. The error is already displayed.
      if (!chatSessionRef.current) {
        return;
      }
      
      setMessages([]);
      setIsLoading(true);
      setError(null); // Clear previous fetch errors.

      const prompt = `Tell me about the county with FIPS code: ${selectedCounty.id}.`;

      chatSessionRef.current.sendMessage({ message: prompt })
        .then(response => {
          setMessages([
            { author: 'system', content: `Exploring county FIPS: ${selectedCounty.id}` },
            { author: 'bot', content: response.text }
          ]);
        })
        .catch(err => {
          console.error("Gemini API error:", err);
          const errorMsg = "Sorry, I couldn't fetch information for this county.";
          setError(errorMsg);
          onError(`Gemini Fetch Error: ${err instanceof Error ? err.message : String(err)}`);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [selectedCounty, onError]);

  // Scroll to the bottom of the message list
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);


  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading || !chatSessionRef.current) return;

    const userMessage: ChatMessage = { author: 'user', content: userInput };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await chatSessionRef.current.sendMessage({ message: userInput });
      setMessages(prev => [...prev, { author: 'bot', content: response.text }]);
    } catch (err) {
      console.error("Gemini API send error:", err);
      const errorMsg = "Error sending message. Please try again.";
      setError(errorMsg);
      setMessages(prev => [...prev, { author: 'system', content: errorMsg }]);
      onError(`Gemini Send Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-gray-800 text-white">
      <h2 className="text-lg font-bold text-cyan-400 p-4 border-b border-cyan-500/30">AI County Assistant</h2>
      <div className="flex-1 p-4 overflow-y-auto space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.author === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs md:max-w-sm lg:max-w-md p-3 rounded-lg ${
                msg.author === 'user' ? 'bg-cyan-600' :
                msg.author === 'bot' ? 'bg-gray-700' : 'bg-transparent text-center w-full text-gray-400 text-sm'
            }`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="flex justify-start">
                <div className="bg-gray-700 p-3 rounded-lg">
                    <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                    </div>
                </div>
            </div>
        )}
         {error && (
            <div className="bg-red-500/20 text-red-300 p-3 rounded-lg text-center">
                {error}
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSendMessage} className="p-4 border-t border-cyan-500/30">
        <div className="flex items-center bg-gray-700 rounded-lg">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={!selectedCounty ? "Select a county first" : "Ask a follow-up question..."}
            className="flex-grow bg-transparent p-3 text-white placeholder-gray-400 focus:outline-none"
            disabled={isLoading || !selectedCounty || !!error}
          />
          <button
            type="submit"
            className="p-3 text-cyan-400 hover:text-cyan-200 disabled:text-gray-500 disabled:cursor-not-allowed"
            disabled={isLoading || !userInput.trim() || !!error}
            aria-label="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" /></svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default Chat;