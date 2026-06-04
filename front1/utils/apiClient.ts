const DEFAULT_API_BASE_URL = 'http://localhost:8002';
const BROWSER_API_KEY_STORAGE = 'mcpchat_api_key';

export const API_BASE_URL = (import.meta.env.VITE_BACKEND_API_BASE_URL || DEFAULT_API_BASE_URL).trim();

function readStoredApiKey(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  return window.localStorage.getItem(BROWSER_API_KEY_STORAGE)?.trim() || '';
}

export function getBackendApiKey(): string {
  return (import.meta.env.VITE_BACKEND_API_KEY || readStoredApiKey() || '').trim();
}

export function buildApiHeaders(baseHeaders: Record<string, string> = {}): Record<string, string> {
  const headers = { ...baseHeaders };
  const apiKey = getBackendApiKey();
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
}
