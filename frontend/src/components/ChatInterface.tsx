import { useState, useEffect } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Send, Bot, User } from 'lucide-react';
import { API_ENDPOINTS } from '../config/api';

const SESSION_STORAGE_KEY = 'dataagent_chat_session_id';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const initialMessages: Message[] = [
  {
    id: '1',
    type: 'assistant',
    content: '您好！我是数据分析助手，可以帮您分析上传的数据集。请在右侧上传CSV文件，然后选择变量进行分析，我会为您提供详细的分析结果和建议。',
    timestamp: new Date()
  }
];

interface ChatInterfaceProps {
  clearTrigger: number;
  onImageGenerated?: (imageUrl: string) => void;
}

export function ChatInterface({ clearTrigger, onImageGenerated }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>(() => {
    if (typeof window === 'undefined') {
      return '';
    }
    return window.localStorage.getItem(SESSION_STORAGE_KEY) || '';
  });
  const syncSessionId = (nextSessionId: string) => {
    setSessionId(nextSessionId);
    if (typeof window === 'undefined') {
      return;
    }

    if (nextSessionId) {
      window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
    } else {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
    }
  };

  useEffect(() => {
    if (clearTrigger > 0) {
      const oldSessionId = sessionId;
      if (oldSessionId) {
        fetch(`${API_ENDPOINTS.CHAT_CLEAR_SESSION}/${oldSessionId}`, {
          method: 'DELETE'
        }).catch((error) => {
          console.warn('Failed to clear backend session:', error);
        });
      }

      syncSessionId('');
      setMessages(initialMessages);
      setInputValue('');
      setIsLoading(false);
    }
  }, [clearTrigger]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages((prev: Message[]) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date()
    };
    
    setMessages((prev: Message[]) => [...prev, assistantMessage]);

    try {
      const requestBody = {
        message: userMessage.content,
        session_id: sessionId || undefined
      };

      const response = await fetch(API_ENDPOINTS.CHAT_STREAM, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let buffer = '';

      const updateAssistantContent = (text: string) => {
        const currentText = text || '正在分析中...';
        const imageMatch = currentText.match(/IMAGE_GENERATED:\s*(\S+)/);
        if (imageMatch && onImageGenerated) {
          const filename = imageMatch[1];
          const imageUrl = `http://localhost:8002/static/images/${filename}`;
          onImageGenerated(imageUrl);
        }

        setMessages((prev: Message[]) => prev.map((msg: Message) => 
          msg.id === assistantMessageId 
            ? { ...msg, content: currentText }
            : msg
        ));
      };

      const handleSSEBlock = (block: string) => {
        const normalizedBlock = block.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        if (!normalizedBlock.trim()) {
          return;
        }

        const lines = normalizedBlock.split('\n');
        let eventType = 'message';
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
          }
        }

        const dataContent = dataLines.join('\n');

        if (eventType === 'chunk') {
          accumulatedContent += dataContent;
          updateAssistantContent(accumulatedContent);
          return;
        }

        if (eventType === 'meta' || eventType === 'done') {
          try {
            const payload = JSON.parse(dataContent);
            if (payload.session_id && typeof payload.session_id === 'string') {
              syncSessionId(payload.session_id);
            }
            if (payload.answer && typeof payload.answer === 'string') {
              accumulatedContent = payload.answer;
              updateAssistantContent(accumulatedContent);
            }
          } catch (error) {
            console.warn('Failed to parse SSE payload:', error);
          }
          return;
        }

        if (eventType === 'error') {
          const errorMessage = dataContent || '流式响应发生错误。';
          updateAssistantContent(errorMessage);
          return;
        }
      };

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const decodedChunk = decoder.decode(value, { stream: true });
          buffer += decodedChunk;
          
          const blocks = buffer.split(/\r?\n\r?\n/);
          if (!/\r?\n\r?\n$/.test(buffer)) {
            buffer = blocks.pop() || '';
          } else {
            buffer = '';
          }
          
          for (const block of blocks) {
            handleSSEBlock(block);
          }
        }
        
        if (buffer.trim()) {
          handleSSEBlock(buffer);
        }
      }

      setIsLoading(false);
    } catch (error) {
      console.error('Error calling agent:', error);
      
      setMessages((prev: Message[]) => prev.map((msg: Message) => 
        msg.id === assistantMessageId 
          ? { ...msg, content: '抱歉，调用AI助手时出现错误。请检查后端服务是否正常运行。' }
          : msg
      ));
      
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-6 custom-scrollbar">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.type === 'assistant' && (
                <div className="w-10 h-10 bg-primary border-2 border-black flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              
              <div
                className={`max-w-[80%] p-4 whitespace-pre-wrap border-2 border-black ${
                  message.type === 'user'
                    ? 'bg-secondary text-black ml-auto'
                    : 'bg-white text-black'
                }`}
              >
                {message.content}
              </div>
              
              {message.type === 'user' && (
                <div className="w-10 h-10 bg-[#FFD166] border-2 border-black flex items-center justify-center flex-shrink-0 overflow-hidden">
                  <img 
                    src="https://api.dicebear.com/7.x/pixel-art/svg?seed=user&backgroundColor=FFD166" 
                    alt="User" 
                    className="w-full h-full"
                  />
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-10 h-10 bg-primary border-2 border-black flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white border-2 border-black p-4">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-primary border border-black"></div>
                  <div className="w-3 h-3 bg-secondary border border-black" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-3 h-3 bg-accent border border-black" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          )}
        </div>
        </ScrollArea>
      </div>

      <div className="p-6 border-t-2 border-black">
        <div className="flex gap-3">
          <Textarea
            value={inputValue}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="请输入您的问题..."
            className="flex-1 min-h-[50px] max-h-[120px] resize-none font-medium"
            disabled={isLoading}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="self-end"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}