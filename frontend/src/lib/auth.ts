import { API_ENDPOINTS, apiFetch } from '../config/api';

export interface AuthUser {
  id: string;
  username: string;
  createdAt: string;
}

const normalizeUser = (payload: unknown): AuthUser => {
  const record = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
  return {
    id: typeof record.id === 'string' ? record.id : '',
    username: typeof record.username === 'string' ? record.username : '',
    createdAt: typeof record.created_at === 'string'
      ? record.created_at
      : typeof record.createdAt === 'string'
        ? record.createdAt
        : '',
  };
};

const extractErrorMessage = async (response: Response) => {
  try {
    const payload = await response.json() as { detail?: string };
    if (typeof payload.detail === 'string' && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
  }

  return `HTTP error! status: ${response.status}`;
};

export const fetchCurrentUser = async (): Promise<AuthUser | null> => {
  const response = await apiFetch(API_ENDPOINTS.AUTH_ME);
  if (response.status === 401) {
    return null;
  }
  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  const payload = await response.json() as { user?: unknown };
  if (!payload.user) {
    return null;
  }

  return normalizeUser(payload.user);
};

const submitAuth = async (endpoint: string, username: string, password: string) => {
  const response = await apiFetch(endpoint, {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  const payload = await response.json() as { user?: unknown };
  return normalizeUser(payload.user);
};

export const login = async (username: string, password: string) => {
  return submitAuth(API_ENDPOINTS.AUTH_LOGIN, username, password);
};

export const register = async (username: string, password: string) => {
  return submitAuth(API_ENDPOINTS.AUTH_REGISTER, username, password);
};

export const logout = async () => {
  const response = await apiFetch(API_ENDPOINTS.AUTH_LOGOUT, {
    method: 'POST',
  });

  if (!response.ok && response.status !== 401) {
    throw new Error(await extractErrorMessage(response));
  }
};
