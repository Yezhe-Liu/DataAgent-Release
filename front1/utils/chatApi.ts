import { Conversation, Message, RetrievalBlock, TextBlock, ToolCallBlock, WorkflowTraceBlock } from '../types';
import { API_BASE_URL, buildApiHeaders } from './apiClient';

export interface SessionItem {
  id: string;
  title: string;
  updated_at: number;
  created_at?: number;
}

export interface HistorySessionPayload {
  session: {
    id: string;
    title: string;
    created_at: number;
    updated_at: number;
  };
  settings?: {
    active_kb_id?: string;
    rag_enabled?: boolean;
    top_k_override?: number | null;
  };
  summary?: string;
  memory_notes?: string[];
  messages: Array<Record<string, any>>;
  traces?: Array<{
    timestamp: number;
    route?: string;
    query?: string;
    workflow?: Array<Record<string, any>>;
    tools?: Array<Record<string, any>>;
    retrieval?: Record<string, any>;
  }>;
}

export interface SessionSettingsPayload {
  session_id: string;
  active_kb_id: string;
  rag_enabled: boolean;
  top_k_override?: number | null;
}

type HistoricalMessageBlock = TextBlock | ToolCallBlock | WorkflowTraceBlock | RetrievalBlock;

function normalizeTimestamp(value: unknown, fallback: number = Date.now()): number {
  const numericValue = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return fallback;
  }
  return numericValue < 1_000_000_000_000 ? numericValue * 1000 : numericValue;
}

function extractContent(rawContent: any): string {
  if (typeof rawContent === 'string') {
    return rawContent;
  }

  if (Array.isArray(rawContent)) {
    return rawContent
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          return item.text || item.content || JSON.stringify(item, null, 2);
        }
        return String(item ?? '');
      })
      .join('\n');
  }

  if (rawContent && typeof rawContent === 'object') {
    return rawContent.text || rawContent.content || JSON.stringify(rawContent, null, 2);
  }

  return String(rawContent ?? '');
}

function convertWorkflowToBlocks(workflow: Array<Record<string, any>> = [], timestamp: number): WorkflowTraceBlock[] {
  return workflow.map((step, index) => ({
    id: `workflow-${timestamp}-${index}`,
    type: 'workflow',
    stage: String(step.stage || 'workflow'),
    status: step.status === 'running' ? 'running' : 'completed',
    message: String(step.message || ''),
    route: step.route,
    timestamp: timestamp + index,
  }));
}

function convertRetrievalToBlocks(retrieval: Record<string, any> | undefined, timestamp: number): RetrievalBlock[] {
  const documents = Array.isArray(retrieval?.documents) ? retrieval.documents : [];
  if (documents.length === 0) {
    return [];
  }
  return [{
    id: `retrieval-history-${timestamp}`,
    type: 'retrieval',
    query: String(retrieval?.query || ''),
    strategy: retrieval?.strategy ? String(retrieval.strategy) : undefined,
    documents: documents.map((document: any) => ({
      id: String(document.id || ''),
      title: String(document.title || document.source_file || '未命名文档'),
      category: String(document.category || document.source_type || 'document'),
      score: typeof document.score === 'number' ? document.score : Number(document.score) || 0,
      keyword_score: typeof document.keyword_score === 'number' ? document.keyword_score : undefined,
      vector_score: typeof document.vector_score === 'number' ? document.vector_score : undefined,
      relevance: typeof document.relevance === 'number' ? document.relevance : undefined,
      excerpt: document.excerpt ? String(document.excerpt) : undefined,
      content: document.content ? String(document.content) : undefined,
      source_file: document.source_file ? String(document.source_file) : undefined,
      section_path: document.section_path ? String(document.section_path) : undefined,
      page: typeof document.page === 'number' ? document.page : undefined,
      citation_id: typeof document.citation_id === 'number' ? document.citation_id : undefined,
      metadata: document.metadata && typeof document.metadata === 'object' ? document.metadata : undefined,
    })),
    timestamp,
  }];
}

function convertToolsToBlocks(tools: Array<Record<string, any>> = [], timestamp: number): ToolCallBlock[] {
  return tools.map((tool, index) => ({
    id: `tool-history-${timestamp}-${index}`,
    type: 'tool_call',
    toolName: String(tool.tool_name || tool.name || 'tool'),
    input: tool.input,
    output: tool.output,
    status: tool.status === 'running' ? 'running' : tool.status === 'failed' ? 'failed' : 'completed',
    timestamp: timestamp + index,
    isExpanded: false,
  }));
}

function settleHistoricalWorkflowBlocks(blocks: HistoricalMessageBlock[]): HistoricalMessageBlock[] {
  return blocks.map((block): HistoricalMessageBlock => {
    if (block.type !== 'workflow' || block.status !== 'running') {
      return block;
    }
    return {
      ...block,
      status: 'completed' as const,
    };
  });
}

export function convertSessionItemToConversation(session: SessionItem): Conversation {
  const createdAt = normalizeTimestamp(session.created_at ?? session.updated_at);
  const updatedAt = normalizeTimestamp(session.updated_at, createdAt);
  return {
    id: session.id,
    title: session.title,
    messages: [],
    settings: {
      activeKbId: '',
      ragEnabled: true,
      topKOverride: null,
    },
    createdAt,
    updatedAt,
  };
}

export function convertHistoryToConversation(payload: HistorySessionPayload): Conversation {
  const traces = payload.traces || [];
  const tracePairs = traces.slice();
  let assistantTraceIndex = 0;
  const sessionCreatedAt = normalizeTimestamp(payload.session?.created_at);
  const sessionUpdatedAt = normalizeTimestamp(payload.session?.updated_at, sessionCreatedAt);

  const messages: Message[] = payload.messages.map((rawMessage, index) => {
    const rawType = rawMessage?.type || rawMessage?.data?.type || '';
    const role = rawType === 'human' ? 'user' : 'assistant';
    const content = extractContent(rawMessage?.data?.content);
    const timestamp = sessionCreatedAt + index;

    if (role === 'user') {
      return {
        id: `history-user-${payload.session.id}-${index}`,
        role,
        content,
        timestamp,
        isComplete: true,
      };
    }

    const currentTrace = tracePairs[assistantTraceIndex] || null;
    assistantTraceIndex += 1;
    const traceTimestamp = currentTrace ? normalizeTimestamp(currentTrace.timestamp, timestamp) : timestamp;
    const workflowBlocks = currentTrace ? convertWorkflowToBlocks(currentTrace.workflow, traceTimestamp) : [];
    const toolBlocks = currentTrace ? convertToolsToBlocks(currentTrace.tools, traceTimestamp + workflowBlocks.length + 1) : [];
    const retrievalBlocks = currentTrace ? convertRetrievalToBlocks(currentTrace.retrieval, traceTimestamp + workflowBlocks.length + toolBlocks.length + 1) : [];
    const textBlock: TextBlock = {
      id: `history-text-${payload.session.id}-${index}`,
      type: 'text',
      content,
      timestamp: traceTimestamp + workflowBlocks.length + toolBlocks.length + retrievalBlocks.length + 1,
    };
    const blocks: HistoricalMessageBlock[] = settleHistoricalWorkflowBlocks([
      ...workflowBlocks,
      ...toolBlocks,
      ...retrievalBlocks,
      textBlock,
    ]);

    return {
      id: `history-assistant-${payload.session.id}-${index}`,
      role,
      blocks,
      timestamp,
      isComplete: true,
      isStreaming: false,
    };
  });

  return {
    id: payload.session.id,
    title: payload.session.title,
    messages,
    settings: {
      activeKbId: String(payload.settings?.active_kb_id || ''),
      ragEnabled: payload.settings?.rag_enabled !== false,
      topKOverride: payload.settings?.top_k_override ?? null,
    },
    createdAt: sessionCreatedAt,
    updatedAt: sessionUpdatedAt,
  };
}

export async function getSessions(): Promise<SessionItem[]> {
  const response = await fetch(`${API_BASE_URL}/sessions`, {
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch sessions: ${response.statusText}`);
  }
  return response.json();
}

export async function createSession(): Promise<SessionItem> {
  const response = await fetch(`${API_BASE_URL}/sessions`, {
    method: 'POST',
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.statusText}`);
  }
  return response.json();
}

export async function getHistory(sessionId: string): Promise<HistorySessionPayload> {
  const response = await fetch(`${API_BASE_URL}/history/${encodeURIComponent(sessionId)}`, {
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch history: ${response.statusText}`);
  }
  return response.json();
}

export async function renameSession(sessionId: string, title: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/title`, {
    method: 'PATCH',
    headers: buildApiHeaders({
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error(`Failed to rename session: ${response.statusText}`);
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: buildApiHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
}

export async function getSessionSettings(sessionId: string): Promise<SessionSettingsPayload> {
  const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/settings`, {
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch session settings: ${response.statusText}`);
  }
  return response.json();
}

export async function updateSessionSettings(
  sessionId: string,
  payload: Partial<SessionSettingsPayload>
): Promise<SessionSettingsPayload> {
  const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/settings`, {
    method: 'PATCH',
    headers: buildApiHeaders({
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Failed to update session settings: ${response.statusText}`);
  }
  return response.json();
}
