import { useEffect, useMemo, useState } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { Bot, CheckCircle2, Cpu, Database, Loader2, PanelLeft, PlugZap, Send, Wrench, X } from 'lucide-react';

import { API_ENDPOINTS, apiFetch } from '../config/api';
import {
  buildStoredSessionFromApiDetail,
  buildStoredSessionFromApiSummary,
  buildStoredSession,
  createInitialMessages,
  createSessionId,
  extractGeneratedImageUrl,
  getInitialChatState,
  getSessionMessages,
  persistCurrentSessionId,
  removeStoredSession,
  saveStoredSessions,
  shouldPersistSession,
  upsertStoredSession,
  type ApiSessionDetail,
  type ApiSessionSummary,
  type Message,
  type StoredChatSession,
} from '../lib/chatSessions';
import { SessionSidebar } from './SessionSidebar';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Textarea } from './ui/textarea';

interface ChatWorkspaceProps {
  clearTrigger: number;
  userId: string;
  onImageGenerated?: (imageUrl: string | null) => void;
}

interface StreamStatusItem {
  id: string;
  scope: 'system' | 'model' | 'tool';
  status: 'running' | 'success' | 'error';
  title: string;
  detail: string;
  toolName: string;
  toolOrigin: string;
  timestamp: string;
}

export function ChatWorkspace({ clearTrigger, userId, onImageGenerated }: ChatWorkspaceProps) {
  const initialState = useMemo(() => getInitialChatState(userId), [userId]);
  const [sessions, setSessions] = useState<StoredChatSession[]>(initialState.sessions);
  const [sessionId, setSessionId] = useState(initialState.sessionId);
  const [messages, setMessages] = useState<Message[]>(initialState.messages);
  const [streamStatuses, setStreamStatuses] = useState<StreamStatusItem[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    if (typeof window === 'undefined') {
      return true;
    }

    return window.innerWidth >= 1024;
  });

  useEffect(() => {
    setSessions(initialState.sessions);
    setSessionId(initialState.sessionId);
    setMessages(initialState.messages);
    setStreamStatuses([]);
    setInputValue('');
    setIsLoading(false);
  }, [initialState]);

  const syncSessionId = (nextSessionId: string) => {
    setSessionId(nextSessionId);
    persistCurrentSessionId(nextSessionId, userId);
  };

  const activateSession = (targetSessionId: string, availableSessions: StoredChatSession[]) => {
    syncSessionId(targetSessionId);
    setMessages(getSessionMessages(targetSessionId, availableSessions));
    setStreamStatuses([]);
    setInputValue('');
    setIsLoading(false);
  };

  const upsertStreamStatus = (nextStatus: StreamStatusItem) => {
    setStreamStatuses((previousStatuses) => {
      const matchedIndex = previousStatuses.findIndex((item) => item.id === nextStatus.id);
      if (matchedIndex < 0) {
        return [...previousStatuses, nextStatus];
      }

      const nextStatuses = [...previousStatuses];
      nextStatuses[matchedIndex] = {
        ...nextStatuses[matchedIndex],
        ...nextStatus,
      };
      return nextStatuses;
    });
  };

  const mergeSyncedSessions = (baseSessions: StoredChatSession[], incomingSessions: StoredChatSession[]) => {
    let nextSessions = [...baseSessions];

    for (const incomingSession of incomingSessions) {
      const existingSession = nextSessions.find((session) => session.id === incomingSession.id);
      nextSessions = upsertStoredSession(nextSessions, {
        ...incomingSession,
        messages: incomingSession.messages.length ? incomingSession.messages : existingSession?.messages || [],
      });
    }

    return nextSessions;
  };

  const fetchSessionDetailFromBackend = async (targetSessionId: string, cachedSession?: StoredChatSession) => {
    const response = await apiFetch(`${API_ENDPOINTS.CHAT_SESSION}/${targetSessionId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const payload = await response.json() as { session?: ApiSessionDetail };
    if (!payload.session) {
      throw new Error('Session payload is missing');
    }

    const detailSession = buildStoredSessionFromApiDetail(payload.session);
    return detailSession.messages.length
      ? detailSession
      : {
          ...detailSession,
          messages: cachedSession?.messages || [],
        };
  };

  const hydrateSessionFromBackend = async (
    targetSessionId: string,
    cachedSession?: StoredChatSession,
    shouldActivate = false,
  ) => {
    try {
      const detailSession = await fetchSessionDetailFromBackend(targetSessionId, cachedSession);
      setSessions((previousSessions) => upsertStoredSession(previousSessions, detailSession));

      if (shouldActivate) {
        activateSession(targetSessionId, [detailSession]);
      }

      return detailSession;
    } catch (error) {
      console.warn('Failed to load backend session detail:', error);
      return null;
    }
  };

  useEffect(() => {
    persistCurrentSessionId(sessionId, userId);
  }, [sessionId, userId]);

  useEffect(() => {
    saveStoredSessions(sessions, userId);
  }, [sessions, userId]);

  useEffect(() => {
    onImageGenerated?.(extractGeneratedImageUrl(messages));
  }, [messages, onImageGenerated]);

  useEffect(() => {
    let isCancelled = false;

    const syncSessionsFromBackend = async () => {
      try {
        const response = await apiFetch(API_ENDPOINTS.CHAT_SESSIONS);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const payload = await response.json() as { sessions?: ApiSessionSummary[] };
        if (!Array.isArray(payload.sessions) || isCancelled) {
          return;
        }

        const remoteSessions = payload.sessions.map(buildStoredSessionFromApiSummary);
        setSessions((previousSessions) => mergeSyncedSessions(previousSessions, remoteSessions));

        const activeRemoteSessionId = initialState.sessionId && remoteSessions.some((session) => session.id === initialState.sessionId)
          ? initialState.sessionId
          : '';

        if (!activeRemoteSessionId) {
          return;
        }

        const cachedActiveSession = sessions.find((session) => session.id === activeRemoteSessionId);
        const detailSession = await fetchSessionDetailFromBackend(activeRemoteSessionId, cachedActiveSession);
        if (isCancelled) {
          return;
        }

        setSessions((previousSessions) => upsertStoredSession(previousSessions, detailSession));
        activateSession(activeRemoteSessionId, [detailSession]);
      } catch (error) {
        console.warn('Failed to sync backend sessions:', error);
      }
    };

    void syncSessionsFromBackend();

    return () => {
      isCancelled = true;
    };
  }, [initialState, userId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    setSessions((previousSessions) => {
      const hasCurrentSession = previousSessions.some((session) => session.id === sessionId);
      if (!shouldPersistSession(messages) && !hasCurrentSession) {
        return previousSessions;
      }

      return upsertStoredSession(previousSessions, buildStoredSession(sessionId, messages));
    });
  }, [messages, sessionId]);

  useEffect(() => {
    if (clearTrigger > 0) {
      void handleDeleteSession(sessionId);
    }
  }, [clearTrigger]);

  const createAndSelectSession = () => {
    const nextSessionId = createSessionId();
    const nextMessages = createInitialMessages();
    const nextSession = buildStoredSession(nextSessionId, nextMessages);

    setSessions((previousSessions) => upsertStoredSession(previousSessions, nextSession));
    syncSessionId(nextSessionId);
    setMessages(nextMessages);
    setStreamStatuses([]);
    setInputValue('');
    setIsLoading(false);
    onImageGenerated?.(null);
  };

  const handleSelectSession = (targetSessionId: string) => {
    if (targetSessionId === sessionId) {
      return;
    }

    const cachedSession = sessions.find((session) => session.id === targetSessionId);
    activateSession(targetSessionId, cachedSession ? [cachedSession] : []);
    void hydrateSessionFromBackend(targetSessionId, cachedSession, true);
  };

  const handleDeleteSession = async (targetSessionId: string) => {
    if (!targetSessionId) {
      createAndSelectSession();
      return;
    }

    apiFetch(`${API_ENDPOINTS.CHAT_CLEAR_SESSION}/${targetSessionId}`, {
      method: 'DELETE',
    }).catch((error) => {
      console.warn('Failed to clear backend session:', error);
    });

    const nextSessions = removeStoredSession(sessions, targetSessionId);

    if (!nextSessions.length) {
      const nextSessionId = createSessionId();
      const nextMessages = createInitialMessages();
      const nextSession = buildStoredSession(nextSessionId, nextMessages);
      setSessions([nextSession]);
      syncSessionId(nextSessionId);
      setMessages(nextMessages);
      setInputValue('');
      setIsLoading(false);
      onImageGenerated?.(null);
      return;
    }

    setSessions(nextSessions);

    if (targetSessionId === sessionId) {
      const nextActiveSessionId = nextSessions[0].id;
      const cachedSession = nextSessions.find((session) => session.id === nextActiveSessionId);
      activateSession(nextActiveSessionId, nextSessions);
      await hydrateSessionFromBackend(nextActiveSessionId, cachedSession, true);
      return;
    }

    if (!nextSessions.some((session) => session.id === sessionId)) {
      const nextActiveSessionId = nextSessions[0].id;
      const cachedSession = nextSessions.find((session) => session.id === nextActiveSessionId);
      activateSession(nextActiveSessionId, nextSessions);
      await hydrateSessionFromBackend(nextActiveSessionId, cachedSession, true);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) {
      return;
    }

    const activeSessionId = sessionId || createSessionId();
    if (!sessionId) {
      syncSessionId(activeSessionId);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    const assistantMessageId = `${Date.now() + 1}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    setMessages((previousMessages) => [...previousMessages, userMessage, assistantMessage]);
    setStreamStatuses([]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await apiFetch(API_ENDPOINTS.CHAT_STREAM, {
        method: 'POST',
        body: JSON.stringify({
          message: userMessage.content,
          session_id: activeSessionId,
        }),
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
        setMessages((previousMessages) => previousMessages.map((message) => {
          if (message.id !== assistantMessageId) {
            return message;
          }

          return {
            ...message,
            content: currentText,
          };
        }));
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
            const rawData = line.slice(5);
            dataLines.push(rawData.startsWith(' ') ? rawData.slice(1) : rawData);
          }
        }

        const dataContent = dataLines.join('\n');

        if (eventType === 'chunk') {
          accumulatedContent += dataContent;
          updateAssistantContent(accumulatedContent);
          return;
        }

        if (eventType === 'status') {
          try {
            const payload = JSON.parse(dataContent) as Record<string, unknown>;
            if (typeof payload.id !== 'string' || typeof payload.title !== 'string') {
              return;
            }

            upsertStreamStatus({
              id: payload.id,
              scope: payload.scope === 'model' || payload.scope === 'tool' ? payload.scope : 'system',
              status: payload.status === 'success' || payload.status === 'error' ? payload.status : 'running',
              title: payload.title,
              detail: typeof payload.detail === 'string' ? payload.detail : '',
              toolName: typeof payload.tool_name === 'string'
                ? payload.tool_name
                : typeof payload.toolName === 'string'
                  ? payload.toolName
                  : '',
              toolOrigin: typeof payload.tool_origin === 'string'
                ? payload.tool_origin
                : typeof payload.toolOrigin === 'string'
                  ? payload.toolOrigin
                  : '',
              timestamp: typeof payload.timestamp === 'string' ? payload.timestamp : new Date().toISOString(),
            });
          } catch (error) {
            console.warn('Failed to parse SSE status payload:', error);
          }
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
          updateAssistantContent(dataContent || '流式响应发生错误。');
        }
      };

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
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
    } catch (error) {
      console.error('Error calling agent:', error);
      upsertStreamStatus({
        id: `network-error-${assistantMessageId}`,
        scope: 'system',
        status: 'error',
        title: '请求失败，请检查后端服务状态',
        detail: error instanceof Error ? error.message : '未知网络错误',
        toolName: '',
        toolOrigin: '',
        timestamp: new Date().toISOString(),
      });
      setMessages((previousMessages) => previousMessages.map((message) => {
        if (message.id !== assistantMessageId) {
          return message;
        }

        return {
          ...message,
          content: '抱歉，调用AI助手时出现错误。请检查后端服务是否正常运行。',
        };
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleSendMessage();
    }
  };

  const activeSession = sessions.find((session) => session.id === sessionId);
  const sidebarColumnWidth = isSidebarOpen ? 'clamp(260px, 28vw, 320px)' : '0px';
  const latestMessage = messages[messages.length - 1];
  const isWaitingForFirstChunk = isLoading && !(
    latestMessage?.type === 'assistant' && latestMessage.content.trim().length > 0
  );
  const getStatusScopeIcon = (statusItem: StreamStatusItem) => {
    if (statusItem.toolOrigin === 'knowledge_base') {
      return <Database className="h-4 w-4 text-[#946200]" />;
    }
    if (statusItem.toolOrigin === 'mcp') {
      return <PlugZap className="h-4 w-4 text-[#6b46c1]" />;
    }
    if (statusItem.scope === 'model') {
      return <Cpu className="h-4 w-4 text-[#1d4ed8]" />;
    }
    if (statusItem.scope === 'tool') {
      return <Wrench className="h-4 w-4 text-[#0f766e]" />;
    }
    return <Bot className="h-4 w-4 text-[#b9770e]" />;
  };
  const getStatusBadge = (status: StreamStatusItem['status']) => {
    if (status === 'success') {
      return {
        className: 'border-[#d8ead7] bg-[#f3fbf2] text-[#2f6f38]',
        label: '已完成',
      };
    }
    if (status === 'error') {
      return {
        className: 'border-[#f3d3d3] bg-[#fff3f3] text-[#b42318]',
        label: '失败',
      };
    }
    return {
      className: 'border-[#f0e1b6] bg-[#fff8e6] text-[#946200]',
      label: '进行中',
    };
  };
  const getStatusOriginLabel = (statusItem: StreamStatusItem) => {
    if (statusItem.toolOrigin === 'knowledge_base') {
      return '知识库/数据库';
    }
    if (statusItem.toolOrigin === 'mcp') {
      return 'MCP';
    }
    if (statusItem.scope === 'model') {
      return '模型';
    }
    if (statusItem.scope === 'tool') {
      return '本地工具';
    }
    return '系统';
  };

  return (
    <div
      className="grid h-full min-h-0 overflow-hidden bg-[#fcfbf8]"
      style={{
        gridTemplateColumns: `${sidebarColumnWidth} minmax(0, 1fr)`,
        transition: 'grid-template-columns 220ms ease',
      }}
    >
      <div
        className={`min-w-0 overflow-hidden bg-[#faf8f3] transition-opacity duration-150 ${
          isSidebarOpen ? 'border-r border-[#e7dfd1] opacity-100' : 'pointer-events-none opacity-0'
        }`}
      >
        <div className="flex h-full min-h-0 flex-col">
          <div className="flex items-center justify-between border-b border-[#e7dfd1] bg-[#fcfbf8] px-4 py-3">
            <div>
              <div className="text-sm font-semibold text-black">会话记录</div>
              <div className="mt-1 text-xs text-black/50">像 ChatGPT 一样管理历史对话</div>
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="rounded-xl border-[#ded4c4] bg-white"
              onClick={() => setIsSidebarOpen(false)}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          <SessionSidebar
            sessions={sessions}
            activeSessionId={sessionId}
            onCreateSession={createAndSelectSession}
            onSelectSession={handleSelectSession}
            onDeleteSession={handleDeleteSession}
            disabled={isLoading}
            className="w-full"
          />
        </div>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-[#fcfbf8]">
        <div className="shrink-0 border-b border-[#e7dfd1] bg-[#fcfbf8] px-4 py-4 sm:px-6">
          <div className="mx-auto flex w-full max-w-4xl items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <Button
                type="button"
                variant="outline"
                className="rounded-xl border-[#ded4c4] bg-white"
                aria-expanded={isSidebarOpen}
                onClick={() => setIsSidebarOpen((previous) => !previous)}
              >
                <PanelLeft className="w-4 h-4" />
                {isSidebarOpen ? '收起会话' : '展开会话'}
              </Button>

              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-black">{activeSession?.title || '当前会话'}</div>
                <div className="mt-1 truncate text-xs text-black/50">
                  {activeSession?.preview || sessionId || '尚未创建会话'}
                </div>
              </div>
            </div>

            <div className="hidden shrink-0 rounded-full border border-[#e7dfd1] bg-white px-3 py-1 text-xs font-medium text-black/55 sm:block">
              历史会话 {sessions.length}
            </div>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full custom-scrollbar">
            <div className="mx-auto flex w-full max-w-3xl flex-col space-y-5 px-4 py-6 sm:px-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.type === 'assistant' && (
                    <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-2xl border border-[#e7dfd1] bg-white shadow-sm">
                      <Bot className="w-5 h-5 text-[#b9770e]" />
                    </div>
                  )}

                  <div
                    className={`max-w-[88%] whitespace-pre-wrap rounded-3xl px-4 py-3 text-[15px] leading-7 shadow-sm lg:max-w-[80%] ${
                      message.type === 'user'
                        ? 'ml-auto border border-[#f0c677] bg-[#fff4dc] text-black'
                        : 'border border-[#e7dfd1] bg-white text-black'
                    }`}
                  >
                    {message.content}
                  </div>

                  {message.type === 'user' && (
                    <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-[#f0c677] bg-[#fff4dc]">
                      <img
                        src="https://api.dicebear.com/7.x/pixel-art/svg?seed=user&backgroundColor=FFD166"
                        alt="User"
                        className="w-full h-full"
                      />
                    </div>
                  )}
                </div>
              ))}

              {streamStatuses.length > 0 && (
                <div className="rounded-[28px] border border-[#e7dfd1] bg-[#fffdf8] p-4 shadow-sm">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-black/45">执行状态</div>
                    <div className="text-xs text-black/45">实时工具与模型调用</div>
                  </div>

                  <div className="mt-3 space-y-2">
                    {streamStatuses.map((statusItem) => {
                      const badge = getStatusBadge(statusItem.status);
                      return (
                        <div key={statusItem.id} className="rounded-2xl border border-[#eee4d6] bg-white/90 px-3 py-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex min-w-0 items-start gap-3">
                              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-[#efe6d7] bg-[#fcfbf8]">
                                {getStatusScopeIcon(statusItem)}
                              </div>

                              <div className="min-w-0">
                                <div className="text-sm font-medium text-black">{statusItem.title}</div>
                                <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.16em] text-black/35">
                                  {getStatusOriginLabel(statusItem)}
                                </div>
                                {statusItem.detail && (
                                  <div className="mt-1 text-xs leading-5 text-black/55">{statusItem.detail}</div>
                                )}
                              </div>
                            </div>

                            <div className={`inline-flex shrink-0 items-center gap-1 rounded-full border px-2 py-1 text-[11px] font-semibold ${badge.className}`}>
                              {statusItem.status === 'running' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                              {badge.label}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {isWaitingForFirstChunk && (
                <div className="flex gap-3 justify-start">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-2xl border border-[#e7dfd1] bg-white shadow-sm">
                    <Bot className="w-5 h-5 text-[#b9770e]" />
                  </div>
                  <div className="rounded-3xl border border-[#e7dfd1] bg-white px-4 py-3 shadow-sm">
                    <div className="flex space-x-2">
                      <div className="h-3 w-3 rounded-full bg-[#f4bf5d]" />
                      <div className="h-3 w-3 rounded-full bg-[#f8d98a]" style={{ animationDelay: '0.1s' }} />
                      <div className="h-3 w-3 rounded-full bg-[#ffd166]" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        <div className="shrink-0 border-t border-[#e7dfd1] bg-white/90 p-4 backdrop-blur-sm sm:p-6">
          <div className="mx-auto flex w-full max-w-3xl items-end gap-3 rounded-[28px] border border-[#e7dfd1] bg-white p-2 shadow-sm">
            <Textarea
              value={inputValue}
              onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setInputValue(event.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="给 Data Agent 发送消息..."
              className="min-h-[52px] max-h-[140px] flex-1 resize-none border-0 bg-transparent px-3 py-2 font-medium shadow-none focus-visible:ring-0"
              disabled={isLoading}
            />
            <Button
              onClick={() => void handleSendMessage()}
              disabled={!inputValue.trim() || isLoading}
              className="h-11 rounded-2xl border-[#f0c677] bg-[#ffd166] px-4 text-black hover:bg-[#ffc94a]"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
