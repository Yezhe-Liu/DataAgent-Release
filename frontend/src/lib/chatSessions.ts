import { API_BASE_URL } from '../config/api';

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface StoredMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface StoredChatSession {
  id: string;
  title: string;
  preview: string;
  updatedAt: string;
  messages: StoredMessage[];
}

export interface ApiSessionSummary {
  session_id: string;
  title: string;
  preview: string;
  updated_at: string;
}

export interface ApiSessionDetail extends ApiSessionSummary {
  messages: Array<{
    type: 'user' | 'assistant';
    content: string;
    timestamp: string;
  }>;
}

export const SESSION_STORAGE_KEY = 'dataagent_chat_session_id';
export const SESSION_LIST_STORAGE_KEY = 'dataagent_chat_sessions';

const INITIAL_ASSISTANT_CONTENT = '您好！我是数据分析助手，可以帮您分析上传的数据集。请在右侧上传CSV文件，然后选择变量进行分析，我会为您提供详细的分析结果和建议。';

const hasWindow = () => typeof window !== 'undefined';

const resolveStorageScope = (userId?: string) => userId?.trim() || '__guest__';

const getSessionStorageKey = (userId?: string) => `${SESSION_STORAGE_KEY}:${resolveStorageScope(userId)}`;

const getSessionListStorageKey = (userId?: string) => `${SESSION_LIST_STORAGE_KEY}:${resolveStorageScope(userId)}`;

const truncateText = (value: string, maxLength: number) => {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}…`;
};

const sanitizeMessageText = (value: string) => value.replace(/IMAGE_GENERATED:\s*\S+/g, '').trim();

const toStoredMessage = (message: Message): StoredMessage => ({
  id: message.id,
  type: message.type,
  content: message.content,
  timestamp: message.timestamp.toISOString(),
});

const toRuntimeMessage = (message: StoredMessage): Message => ({
  id: message.id,
  type: message.type,
  content: message.content,
  timestamp: new Date(message.timestamp),
});

const sortSessions = (sessions: StoredChatSession[]) => {
  return [...sessions].sort((left, right) => {
    return new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime();
  });
};

export const createSessionId = () => {
  return `sess-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
};

export const createInitialMessages = (): Message[] => {
  return [
    {
      id: `welcome-${Date.now().toString(36)}`,
      type: 'assistant',
      content: INITIAL_ASSISTANT_CONTENT,
      timestamp: new Date(),
    },
  ];
};

const toStoredApiMessage = (
  sessionId: string,
  message: ApiSessionDetail['messages'][number],
  index: number,
): StoredMessage => ({
  id: `${sessionId}-msg-${index}`,
  type: message.type,
  content: message.content,
  timestamp: message.timestamp,
});

export const buildStoredSessionFromApiSummary = (session: ApiSessionSummary): StoredChatSession => ({
  id: session.session_id,
  title: session.title || '新对话',
  preview: session.preview || '',
  updatedAt: session.updated_at || new Date().toISOString(),
  messages: [],
});

export const buildStoredSessionFromApiDetail = (session: ApiSessionDetail): StoredChatSession => ({
  id: session.session_id,
  title: session.title || '新对话',
  preview: session.preview || '',
  updatedAt: session.updated_at || new Date().toISOString(),
  messages: Array.isArray(session.messages)
    ? session.messages.map((message, index) => toStoredApiMessage(session.session_id, message, index))
    : [],
});

export const loadStoredSessions = (userId?: string): StoredChatSession[] => {
  if (!hasWindow()) {
    return [];
  }

  try {
    const rawValue = window.localStorage.getItem(getSessionListStorageKey(userId));
    if (!rawValue) {
      return [];
    }

    const parsed = JSON.parse(rawValue);
    if (!Array.isArray(parsed)) {
      return [];
    }

    const normalized = parsed.flatMap((item): StoredChatSession[] => {
      if (!item || typeof item !== 'object') {
        return [];
      }

      const id = typeof item.id === 'string' ? item.id : '';
      const title = typeof item.title === 'string' ? item.title : '新对话';
      const preview = typeof item.preview === 'string' ? item.preview : '';
      const updatedAt = typeof item.updatedAt === 'string' ? item.updatedAt : new Date().toISOString();
      const messages = Array.isArray(item.messages)
        ? item.messages.flatMap((message: unknown): StoredMessage[] => {
            if (!message || typeof message !== 'object') {
              return [];
            }

            const messageRecord = message as Record<string, unknown>;

            const type = messageRecord.type === 'assistant' ? 'assistant' : messageRecord.type === 'user' ? 'user' : null;
            const content = typeof messageRecord.content === 'string' ? messageRecord.content : '';
            const timestamp = typeof messageRecord.timestamp === 'string' ? messageRecord.timestamp : new Date().toISOString();
            const messageId = typeof messageRecord.id === 'string' ? messageRecord.id : `msg-${Date.now().toString(36)}`;

            if (!type) {
              return [];
            }

            return [{
              id: messageId,
              type,
              content,
              timestamp,
            }];
          })
        : [];

      if (!id) {
        return [];
      }

      return [{
        id,
        title,
        preview,
        updatedAt,
        messages,
      }];
    });

    return sortSessions(normalized);
  } catch (error) {
    console.warn('Failed to load stored chat sessions:', error);
    return [];
  }
};

export const saveStoredSessions = (sessions: StoredChatSession[], userId?: string) => {
  if (!hasWindow()) {
    return;
  }

  window.localStorage.setItem(getSessionListStorageKey(userId), JSON.stringify(sortSessions(sessions)));
};

export const getStoredCurrentSessionId = (userId?: string) => {
  if (!hasWindow()) {
    return '';
  }

  return window.localStorage.getItem(getSessionStorageKey(userId)) || '';
};

export const persistCurrentSessionId = (sessionId: string, userId?: string) => {
  if (!hasWindow()) {
    return;
  }

  if (sessionId) {
    window.localStorage.setItem(getSessionStorageKey(userId), sessionId);
    return;
  }

  window.localStorage.removeItem(getSessionStorageKey(userId));
};

export const getSessionMessages = (sessionId: string, sessions: StoredChatSession[]) => {
  const matched = sessions.find((session) => session.id === sessionId);
  if (!matched || !matched.messages.length) {
    return createInitialMessages();
  }

  return matched.messages.map(toRuntimeMessage);
};

export const shouldPersistSession = (messages: Message[]) => {
  return messages.some((message) => message.type === 'user' && sanitizeMessageText(message.content));
};

export const buildStoredSession = (sessionId: string, messages: Message[]): StoredChatSession => {
  const storedMessages = messages.map(toStoredMessage);
  const firstUserMessage = messages.find((message) => message.type === 'user' && sanitizeMessageText(message.content));
  const lastMeaningfulMessage = [...messages].reverse().find((message) => sanitizeMessageText(message.content));
  const title = truncateText(sanitizeMessageText(firstUserMessage?.content || '新对话') || '新对话', 24);
  const previewSource = sanitizeMessageText(lastMeaningfulMessage?.content || firstUserMessage?.content || INITIAL_ASSISTANT_CONTENT) || '暂无内容';
  const updatedAt = lastMeaningfulMessage?.timestamp.toISOString() || new Date().toISOString();

  return {
    id: sessionId,
    title,
    preview: truncateText(previewSource, 40),
    updatedAt,
    messages: storedMessages,
  };
};

export const upsertStoredSession = (sessions: StoredChatSession[], nextSession: StoredChatSession) => {
  const remaining = sessions.filter((session) => session.id !== nextSession.id);
  return sortSessions([nextSession, ...remaining]);
};

export const removeStoredSession = (sessions: StoredChatSession[], sessionId: string) => {
  return sessions.filter((session) => session.id !== sessionId);
};

export const getInitialChatState = (userId?: string) => {
  const sessions = loadStoredSessions(userId);
  const storedCurrentSessionId = getStoredCurrentSessionId(userId);
  const matchedSessionId = storedCurrentSessionId && sessions.some((session) => session.id === storedCurrentSessionId)
    ? storedCurrentSessionId
    : sessions[0]?.id || createSessionId();

  return {
    sessions,
    sessionId: matchedSessionId,
    messages: sessions.some((session) => session.id === matchedSessionId)
      ? getSessionMessages(matchedSessionId, sessions)
      : createInitialMessages(),
  };
};

export const extractGeneratedImageUrl = (messages: Message[]) => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.type !== 'assistant') {
      continue;
    }

    const match = message.content.match(/IMAGE_GENERATED:\s*(\S+)/);
    if (match) {
      return `${API_BASE_URL}/static/images/${match[1]}`;
    }
  }

  return null;
};

export const formatSessionTime = (value: string) => {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return '';
  }

  const now = new Date();
  const isSameDay = now.toDateString() === timestamp.toDateString();
  if (isSameDay) {
    return timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  }

  if (now.getFullYear() === timestamp.getFullYear()) {
    return timestamp.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
  }

  return timestamp.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
};
