// LLM profile 配置 API 客户端
import { API_BASE_URL, buildApiHeaders } from './apiClient';

export interface LLMProfile {
  name: string;
  provider: string;
  model: string;
  streaming: boolean;
  api_key_env: string;
  api_key_available: boolean;
  settings: Record<string, any>;
}

export interface LLMConfigPayload {
  default_profile: string;
  profile_overrides: Record<string, string>;
  profiles: LLMProfile[];
}

export interface LLMConfigUpdate {
  default_profile: string;
  profile_overrides?: Record<string, string>;
}

export async function getLLMConfig(): Promise<LLMConfigPayload> {
  const response = await fetch(`${API_BASE_URL}/config/llm`, {
    method: 'GET',
    headers: buildApiHeaders(),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`获取 LLM 配置失败 (${response.status}): ${detail || response.statusText}`);
  }

  return response.json();
}

export async function updateLLMConfig(payload: LLMConfigUpdate): Promise<LLMConfigPayload> {
  const response = await fetch(`${API_BASE_URL}/config/llm`, {
    method: 'POST',
    headers: buildApiHeaders({
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`更新 LLM 配置失败 (${response.status}): ${detail || response.statusText}`);
  }

  return response.json();
}
