// API 配置
export const API_BASE_URL = 'http://localhost:8002';

// API 端点
export const API_ENDPOINTS = {
  AUTH_REGISTER: `${API_BASE_URL}/auth/register`,
  AUTH_LOGIN: `${API_BASE_URL}/auth/login`,
  AUTH_LOGOUT: `${API_BASE_URL}/auth/logout`,
  AUTH_ME: `${API_BASE_URL}/auth/me`,
  UPLOAD: `${API_BASE_URL}/upload`,
  DATA_PREVIEW: `${API_BASE_URL}/data-preview`,
  CHAT_STREAM: `${API_BASE_URL}/chat/stream`,
  CHAT_INVOKE: `${API_BASE_URL}/chat/invoke`,
  CHAT_SESSIONS: `${API_BASE_URL}/chat/sessions`,
  CHAT_SESSION: `${API_BASE_URL}/chat/session`,
  CHAT_CLEAR_SESSION: `${API_BASE_URL}/chat/session`,
  KB_STATS: `${API_BASE_URL}/kb/stats`,
  KB_REBUILD: `${API_BASE_URL}/kb/rebuild`,
  AGENT_STREAM: `${API_BASE_URL}/agent/stream`,
  AGENT_INVOKE: `${API_BASE_URL}/agent/invoke`,
};

export const apiFetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
  const headers = new Headers(init.headers);
  const isFormDataBody = typeof FormData !== 'undefined' && init.body instanceof FormData;

  if (init.body && !isFormDataBody && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  return fetch(input, {
    ...init,
    credentials: 'include',
    headers,
  });
};
