import { API_BASE_URL, buildApiHeaders } from './apiClient';

export interface UploadedFileRef {
  upload_id: string;
  filename: string;
  original_filename: string;
  size_bytes: number;
  content_type: string;
  uploaded_at: number;
}

export interface RagUploadResponse {
  files: UploadedFileRef[];
  message: string;
}

export interface KnowledgeBaseVersionInfo {
  id: string;
  created_at: number;
  status: 'ready' | 'building' | 'failed';
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  embedding_profile: string;
  file_count: number;
  chunk_count: number;
  topics: string[];
}

export interface KnowledgeBaseFileInfo {
  upload_id: string;
  original_filename: string;
  stored_filename: string;
  size_bytes: number;
  content_type: string;
  sha1: string;
}

export interface KnowledgeBaseSummary {
  id: string;
  name: string;
  description: string;
  status: 'ready' | 'building' | 'failed';
  created_at: number;
  updated_at: number;
  active_version_id: string;
  latest_version_id: string;
  default_top_k: number;
  file_count: number;
  chunk_count: number;
}

export interface KnowledgeBaseDetail extends KnowledgeBaseSummary {
  versions: KnowledgeBaseVersionInfo[];
  files: KnowledgeBaseFileInfo[];
}

export interface KnowledgeBaseCreateRequest {
  kb_id?: string;
  name: string;
  description?: string;
  upload_ids: string[];
  chunk_size?: number;
  chunk_overlap?: number;
  top_k?: number;
  embedding_profile?: string;
}

export interface KnowledgeBaseCreateResponse {
  kb: KnowledgeBaseDetail;
  version_id: string;
  total_chunks: number;
  message: string;
}

export interface RagSourceChunk {
  citation_id: number;
  chunk_id: string;
  content: string;
  excerpt: string;
  score: number;
  relevance: number;
  source_file: string;
  source_type?: string;
  title?: string;
  section_path?: string;
  page?: number | null;
  ordinal?: number;
  metadata?: Record<string, any>;
}

export interface RagRecallResponse {
  kb_id: string;
  kb_name: string;
  kb_version_id: string;
  query: string;
  top_k: number;
  results: RagSourceChunk[];
}

export async function uploadRagFiles(files: File[]): Promise<RagUploadResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  const response = await fetch(`${API_BASE_URL}/rag/upload`, {
    method: 'POST',
    headers: buildApiHeaders(),
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to upload files: ${response.statusText}`);
  }
  return response.json();
}

export async function listKnowledgeBases(): Promise<KnowledgeBaseSummary[]> {
  const response = await fetch(`${API_BASE_URL}/rag/kbs`, {
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to list knowledge bases: ${response.statusText}`);
  }
  return response.json();
}

export async function getKnowledgeBase(kbId: string): Promise<KnowledgeBaseDetail> {
  const response = await fetch(`${API_BASE_URL}/rag/kbs/${encodeURIComponent(kbId)}`, {
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch knowledge base: ${response.statusText}`);
  }
  return response.json();
}

export async function createKnowledgeBase(payload: KnowledgeBaseCreateRequest): Promise<KnowledgeBaseCreateResponse> {
  const response = await fetch(`${API_BASE_URL}/rag/kbs`, {
    method: 'POST',
    headers: buildApiHeaders({
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to create knowledge base: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteKnowledgeBase(kbId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/rag/kbs/${encodeURIComponent(kbId)}`, {
    method: 'DELETE',
    headers: buildApiHeaders(),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to delete knowledge base: ${response.statusText}`);
  }
}

export async function recallKnowledgeBase(kbId: string, query: string, topK?: number): Promise<RagRecallResponse> {
  const response = await fetch(`${API_BASE_URL}/rag/kbs/${encodeURIComponent(kbId)}/recall`, {
    method: 'POST',
    headers: buildApiHeaders({
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to recall knowledge base: ${response.statusText}`);
  }
  return response.json();
}
