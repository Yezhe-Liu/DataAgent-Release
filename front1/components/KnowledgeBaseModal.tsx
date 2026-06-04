import { ChangeEvent, useEffect, useMemo, useState } from 'react';
import { Database, FileText, Files, Link2, Loader2, RefreshCw, Search, Trash2, Unlink2, Upload, X } from 'lucide-react';
import { ConfirmDialog } from './ConfirmDialog';
import { KnowledgeBaseDetail, KnowledgeBaseSummary, RagRecallResponse } from '../utils/ragApi';

interface KnowledgeBaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  knowledgeBases: KnowledgeBaseSummary[];
  currentKnowledgeBase: KnowledgeBaseDetail | null;
  currentKnowledgeBaseId: string;
  activeKbId: string;
  ragEnabled: boolean;
  topKOverride: number | null;
  loadingList: boolean;
  loadingDetail: boolean;
  savingSession: boolean;
  creatingKnowledgeBase: boolean;
  deletingKnowledgeBase: boolean;
  onRefreshKnowledgeBases: () => Promise<void> | void;
  onSelectKnowledgeBase: (kbId: string) => void;
  onSaveSessionSettings: (payload: { activeKbId: string; ragEnabled: boolean; topKOverride: number | null }) => Promise<void>;
  onCreateKnowledgeBase: (payload: {
    name: string;
    description: string;
    files: File[];
    chunkSize?: number;
    chunkOverlap?: number;
    topK?: number;
  }) => Promise<void>;
  onDeleteKnowledgeBase: (kbId: string) => Promise<void>;
  onRecallKnowledgeBase: (kbId: string, query: string, topK?: number) => Promise<RagRecallResponse>;
}

const statusLabelMap: Record<KnowledgeBaseSummary['status'], string> = {
  ready: '可用',
  building: '构建中',
  failed: '失败',
};

const statusClassMap: Record<KnowledgeBaseSummary['status'], string> = {
  ready: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  building: 'border-amber-200 bg-amber-50 text-amber-700',
  failed: 'border-red-200 bg-red-50 text-red-700',
};

function normalizeTimestamp(value: number): number {
  return value < 1_000_000_000_000 ? value * 1000 : value;
}

function formatTimestamp(value: number): string {
  if (!value) return '—';
  return new Date(normalizeTimestamp(value)).toLocaleString('zh-CN');
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let next = value;
  let index = 0;
  while (next >= 1024 && index < units.length - 1) {
    next /= 1024;
    index += 1;
  }
  return `${next.toFixed(next >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

export function KnowledgeBaseModal({
  isOpen,
  onClose,
  knowledgeBases,
  currentKnowledgeBase,
  currentKnowledgeBaseId,
  activeKbId,
  ragEnabled,
  topKOverride,
  loadingList,
  loadingDetail,
  savingSession,
  creatingKnowledgeBase,
  deletingKnowledgeBase,
  onRefreshKnowledgeBases,
  onSelectKnowledgeBase,
  onSaveSessionSettings,
  onCreateKnowledgeBase,
  onDeleteKnowledgeBase,
  onRecallKnowledgeBase,
}: KnowledgeBaseModalProps) {
  const [draftActiveKbId, setDraftActiveKbId] = useState('');
  const [draftRagEnabled, setDraftRagEnabled] = useState(true);
  const [draftTopKOverride, setDraftTopKOverride] = useState('');

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [chunkSize, setChunkSize] = useState('');
  const [chunkOverlap, setChunkOverlap] = useState('');
  const [topK, setTopK] = useState('');

  const [recallQuery, setRecallQuery] = useState('');
  const [recallTopK, setRecallTopK] = useState('');
  const [recallResponse, setRecallResponse] = useState<RagRecallResponse | null>(null);
  const [recallError, setRecallError] = useState('');
  const [isRecalling, setIsRecalling] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setDraftActiveKbId(activeKbId || '');
    setDraftRagEnabled(ragEnabled);
    setDraftTopKOverride(topKOverride != null ? String(topKOverride) : '');
  }, [isOpen, activeKbId, ragEnabled, topKOverride]);

  useEffect(() => {
    if (!isOpen) return;
    setRecallResponse(null);
    setRecallError('');
    setRecallQuery('');
    setRecallTopK('');
  }, [isOpen, currentKnowledgeBaseId]);

  const selectedSummary = useMemo(() => {
    return knowledgeBases.find((item) => item.id === currentKnowledgeBaseId) || null;
  }, [currentKnowledgeBaseId, knowledgeBases]);

  const hasSessionChanges =
    draftActiveKbId !== (activeKbId || '') ||
    draftRagEnabled !== ragEnabled ||
    draftTopKOverride !== (topKOverride != null ? String(topKOverride) : '');

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFiles(Array.from(event.target.files || []));
  };

  const handleSaveSessionSettings = async () => {
    try {
      await onSaveSessionSettings({
        activeKbId: draftActiveKbId,
        ragEnabled: draftRagEnabled,
        topKOverride: draftTopKOverride.trim() ? Number(draftTopKOverride) : null,
      });
    } catch {
      return;
    }
  };

  const handleCreateKnowledgeBase = async () => {
    try {
      await onCreateKnowledgeBase({
        name: name.trim(),
        description: description.trim(),
        files,
        chunkSize: chunkSize.trim() ? Number(chunkSize) : undefined,
        chunkOverlap: chunkOverlap.trim() ? Number(chunkOverlap) : undefined,
        topK: topK.trim() ? Number(topK) : undefined,
      });
      setName('');
      setDescription('');
      setFiles([]);
      setChunkSize('');
      setChunkOverlap('');
      setTopK('');
    } catch {
      return;
    }
  };

  const handleRecall = async () => {
    if (!currentKnowledgeBaseId || !recallQuery.trim()) return;
    setIsRecalling(true);
    setRecallError('');
    try {
      const response = await onRecallKnowledgeBase(
        currentKnowledgeBaseId,
        recallQuery.trim(),
        recallTopK.trim() ? Number(recallTopK) : undefined,
      );
      setRecallResponse(response);
    } catch (error) {
      setRecallResponse(null);
      setRecallError(error instanceof Error ? error.message : '召回测试失败');
    } finally {
      setIsRecalling(false);
    }
  };

  const handleDelete = async () => {
    if (!currentKnowledgeBaseId) return;
    try {
      await onDeleteKnowledgeBase(currentKnowledgeBaseId);
      setShowDeleteConfirm(false);
    } catch {
      return;
    }
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="w-[1180px] max-w-[96vw] max-h-[92vh] bg-white rounded-2xl shadow-2xl overflow-hidden border border-slate-200 flex flex-col">
          <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <div className="text-slate-900 text-lg">知识库管理</div>
                <div className="text-xs text-slate-500">上传文档、创建知识库、绑定当前会话并测试召回。</div>
              </div>
            </div>
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="p-6 overflow-y-auto space-y-6">
            <section className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-sm text-slate-900">当前会话知识库设置</div>
                  <div className="text-xs text-slate-500 mt-1">这里的设置会直接影响当前会话后续的问答检索行为。</div>
                </div>
                <button
                  onClick={handleSaveSessionSettings}
                  disabled={savingSession || !hasSessionChanges}
                  className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  {savingSession ? <Loader2 className="w-4 h-4 animate-spin" /> : <Link2 className="w-4 h-4" />}
                  保存会话设置
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_1fr_180px] gap-3">
                <div className="rounded-xl border border-slate-200 bg-white px-3 py-3">
                  <label className="block text-xs text-slate-500 mb-1">绑定知识库</label>
                  <select
                    value={draftActiveKbId}
                    onChange={(event) => setDraftActiveKbId(event.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                  >
                    <option value="">不绑定知识库</option>
                    {knowledgeBases.map((kb) => (
                      <option key={kb.id} value={kb.id}>
                        {kb.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white px-3 py-3">
                  <label className="block text-xs text-slate-500 mb-1">Top K 覆盖</label>
                  <input
                    value={draftTopKOverride}
                    onChange={(event) => setDraftTopKOverride(event.target.value.replace(/[^0-9]/g, ''))}
                    placeholder="留空使用知识库默认值"
                    className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                  />
                </div>

                <div className="rounded-xl border border-slate-200 bg-white px-3 py-3 flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm text-slate-800">启用检索</div>
                    <div className="text-[11px] text-slate-500 mt-1">关闭后当前会话不走知识库检索。</div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={draftRagEnabled}
                      onChange={(event) => setDraftRagEnabled(event.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-10 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                  </label>
                </div>
              </div>
            </section>

            <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-6">
              <div className="space-y-4">
                <section className="rounded-2xl border border-slate-200 overflow-hidden bg-white">
                  <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                    <div>
                      <div className="text-sm text-slate-900">知识库列表</div>
                      <div className="text-xs text-slate-500 mt-1">选择一个知识库查看详情或做召回测试。</div>
                    </div>
                    <button
                      onClick={() => Promise.resolve(onRefreshKnowledgeBases())}
                      className="w-9 h-9 rounded-lg border border-slate-200 text-slate-500 hover:text-indigo-600 hover:border-indigo-300 flex items-center justify-center transition-colors"
                    >
                      <RefreshCw className={`w-4 h-4 ${loadingList ? 'animate-spin' : ''}`} />
                    </button>
                  </div>

                  <div className="max-h-[340px] overflow-y-auto p-2 space-y-2">
                    {loadingList ? (
                      <div className="py-8 flex items-center justify-center text-sm text-slate-500 gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        正在加载知识库...
                      </div>
                    ) : knowledgeBases.length === 0 ? (
                      <div className="py-8 text-center text-sm text-slate-400">还没有知识库，请先上传文档创建。</div>
                    ) : (
                      knowledgeBases.map((kb) => {
                        const selected = kb.id === currentKnowledgeBaseId;
                        const isActive = kb.id === draftActiveKbId;
                        return (
                          <button
                            key={kb.id}
                            onClick={() => onSelectKnowledgeBase(kb.id)}
                            className={`w-full text-left rounded-xl border px-3 py-3 transition-all ${selected ? 'border-indigo-300 bg-indigo-50/70 shadow-sm' : 'border-slate-200 hover:border-slate-300 bg-white'} `}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0 flex-1">
                                <div className="text-sm text-slate-900 truncate">{kb.name}</div>
                                <div className="text-[11px] text-slate-500 mt-1 truncate">{kb.description || '暂无描述'}</div>
                              </div>
                              <span className={`px-2 py-0.5 rounded-full border text-[10px] ${statusClassMap[kb.status]}`}>
                                {statusLabelMap[kb.status]}
                              </span>
                            </div>
                            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                              <span>{kb.file_count} 文件</span>
                              <span>{kb.chunk_count} 片段</span>
                              {isActive && <span className="text-indigo-600">已绑定当前会话</span>}
                            </div>
                          </button>
                        );
                      })
                    )}
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-100">
                    <div className="text-sm text-slate-900">创建知识库</div>
                    <div className="text-xs text-slate-500 mt-1">支持一次上传多个文件并直接建库。</div>
                  </div>
                  <div className="p-4 space-y-3">
                    <input
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                      placeholder="知识库名称"
                      className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                    />
                    <textarea
                      value={description}
                      onChange={(event) => setDescription(event.target.value)}
                      rows={3}
                      placeholder="知识库描述（可选）"
                      className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400 resize-none"
                    />
                    <label className="flex items-center justify-center gap-2 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-600 hover:border-indigo-300 hover:text-indigo-600 cursor-pointer transition-colors">
                      <Upload className="w-4 h-4" />
                      <span>{files.length > 0 ? `已选择 ${files.length} 个文件` : '选择文档文件'}</span>
                      <input type="file" multiple className="hidden" onChange={handleFileChange} />
                    </label>
                    {files.length > 0 && (
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 max-h-28 overflow-y-auto space-y-1">
                        {files.map((file) => (
                          <div key={`${file.name}-${file.size}`} className="flex items-center justify-between gap-3 text-[11px] text-slate-600">
                            <span className="truncate">{file.name}</span>
                            <span>{formatBytes(file.size)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="grid grid-cols-3 gap-2">
                      <input
                        value={chunkSize}
                        onChange={(event) => setChunkSize(event.target.value.replace(/[^0-9]/g, ''))}
                        placeholder="Chunk Size"
                        className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                      />
                      <input
                        value={chunkOverlap}
                        onChange={(event) => setChunkOverlap(event.target.value.replace(/[^0-9]/g, ''))}
                        placeholder="Overlap"
                        className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                      />
                      <input
                        value={topK}
                        onChange={(event) => setTopK(event.target.value.replace(/[^0-9]/g, ''))}
                        placeholder="Top K"
                        className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                      />
                    </div>
                    <button
                      onClick={handleCreateKnowledgeBase}
                      disabled={creatingKnowledgeBase || !name.trim() || files.length === 0}
                      className="w-full px-4 py-2.5 rounded-lg bg-indigo-600 text-white text-sm hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                    >
                      {creatingKnowledgeBase ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                      创建知识库
                    </button>
                  </div>
                </section>
              </div>

              <div className="space-y-4">
                <section className="rounded-2xl border border-slate-200 bg-white overflow-hidden min-h-[280px]">
                  <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm text-slate-900">知识库详情</div>
                      <div className="text-xs text-slate-500 mt-1">查看版本、文件与召回结果。</div>
                    </div>
                    {currentKnowledgeBaseId && (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setDraftActiveKbId(currentKnowledgeBaseId)}
                          className="px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-700 hover:border-indigo-300 hover:text-indigo-600 transition-colors flex items-center gap-2"
                        >
                          <Link2 className="w-4 h-4" />
                          设为会话知识库
                        </button>
                        <button
                          onClick={() => setShowDeleteConfirm(true)}
                          disabled={deletingKnowledgeBase}
                          className="px-3 py-2 rounded-lg border border-red-200 text-sm text-red-600 hover:bg-red-50 disabled:opacity-60 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                        >
                          {deletingKnowledgeBase ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                          删除
                        </button>
                      </div>
                    )}
                  </div>

                  <div className="p-4 space-y-4">
                    {loadingDetail ? (
                      <div className="py-12 flex items-center justify-center text-sm text-slate-500 gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        正在加载知识库详情...
                      </div>
                    ) : currentKnowledgeBase ? (
                      <>
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                          <div className="flex flex-wrap items-start justify-between gap-4">
                            <div className="min-w-0">
                              <div className="text-base text-slate-900">{currentKnowledgeBase.name}</div>
                              <div className="text-sm text-slate-500 mt-1">{currentKnowledgeBase.description || '暂无描述'}</div>
                            </div>
                            <span className={`px-2.5 py-1 rounded-full border text-xs ${statusClassMap[currentKnowledgeBase.status]}`}>
                              {statusLabelMap[currentKnowledgeBase.status]}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-4 text-xs">
                            <div className="rounded-xl bg-white border border-slate-200 px-3 py-2">
                              <div className="text-slate-500">文件数</div>
                              <div className="text-slate-900 mt-1">{currentKnowledgeBase.file_count}</div>
                            </div>
                            <div className="rounded-xl bg-white border border-slate-200 px-3 py-2">
                              <div className="text-slate-500">片段数</div>
                              <div className="text-slate-900 mt-1">{currentKnowledgeBase.chunk_count}</div>
                            </div>
                            <div className="rounded-xl bg-white border border-slate-200 px-3 py-2">
                              <div className="text-slate-500">当前版本</div>
                              <div className="text-slate-900 mt-1 truncate">{currentKnowledgeBase.active_version_id || '—'}</div>
                            </div>
                            <div className="rounded-xl bg-white border border-slate-200 px-3 py-2">
                              <div className="text-slate-500">更新时间</div>
                              <div className="text-slate-900 mt-1">{formatTimestamp(currentKnowledgeBase.updated_at)}</div>
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 2xl:grid-cols-[1.15fr_1fr] gap-4">
                          <div className="rounded-2xl border border-slate-200 overflow-hidden">
                            <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-2 text-sm text-slate-900">
                              <Files className="w-4 h-4 text-slate-400" />
                              文档文件
                            </div>
                            <div className="max-h-[260px] overflow-y-auto divide-y divide-slate-100">
                              {currentKnowledgeBase.files.length === 0 ? (
                                <div className="px-4 py-6 text-sm text-slate-400">暂无文件记录</div>
                              ) : (
                                currentKnowledgeBase.files.map((file) => (
                                  <div key={file.upload_id || file.original_filename} className="px-4 py-3 flex items-center justify-between gap-3 text-sm">
                                    <div className="min-w-0">
                                      <div className="text-slate-800 truncate">{file.original_filename}</div>
                                      <div className="text-[11px] text-slate-500 mt-1">{file.content_type || '未知类型'}</div>
                                    </div>
                                    <div className="text-[11px] text-slate-500 whitespace-nowrap">{formatBytes(file.size_bytes)}</div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>

                          <div className="rounded-2xl border border-slate-200 overflow-hidden">
                            <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-2 text-sm text-slate-900">
                              <FileText className="w-4 h-4 text-slate-400" />
                              版本信息
                            </div>
                            <div className="max-h-[260px] overflow-y-auto divide-y divide-slate-100">
                              {currentKnowledgeBase.versions.length === 0 ? (
                                <div className="px-4 py-6 text-sm text-slate-400">暂无版本记录</div>
                              ) : (
                                currentKnowledgeBase.versions.map((version) => (
                                  <div key={version.id} className="px-4 py-3 text-sm">
                                    <div className="flex items-center justify-between gap-3">
                                      <div className="text-slate-800 truncate">{version.id}</div>
                                      <span className={`px-2 py-0.5 rounded-full border text-[10px] ${statusClassMap[version.status]}`}>
                                        {statusLabelMap[version.status]}
                                      </span>
                                    </div>
                                    <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-500">
                                      <span>Chunk {version.chunk_size}</span>
                                      <span>Overlap {version.chunk_overlap}</span>
                                      <span>Top K {version.top_k}</span>
                                      <span>{formatTimestamp(version.created_at)}</span>
                                    </div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="rounded-2xl border border-slate-200 overflow-hidden">
                          <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-2 text-sm text-slate-900">
                            <Search className="w-4 h-4 text-slate-400" />
                            召回测试
                          </div>
                          <div className="p-4 space-y-3">
                            <div className="grid grid-cols-1 lg:grid-cols-[1fr_140px_140px] gap-3">
                              <input
                                value={recallQuery}
                                onChange={(event) => setRecallQuery(event.target.value)}
                                placeholder="输入测试问题，例如：售后退款规则是什么？"
                                className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                              />
                              <input
                                value={recallTopK}
                                onChange={(event) => setRecallTopK(event.target.value.replace(/[^0-9]/g, ''))}
                                placeholder="Top K"
                                className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                              />
                              <button
                                onClick={handleRecall}
                                disabled={isRecalling || !recallQuery.trim()}
                                className="px-4 py-2 rounded-lg bg-slate-900 text-white text-sm hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                              >
                                {isRecalling ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                                开始测试
                              </button>
                            </div>

                            {recallError && (
                              <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                                {recallError}
                              </div>
                            )}

                            {recallResponse && (
                              <div className="space-y-3">
                                <div className="text-xs text-slate-500">
                                  命中 {recallResponse.results.length} 条，版本 {recallResponse.kb_version_id || '—'}
                                </div>
                                <div className="space-y-3 max-h-[340px] overflow-y-auto pr-1">
                                  {recallResponse.results.map((result) => (
                                    <div key={result.chunk_id} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                        <span className="rounded bg-indigo-100 text-indigo-700 px-1.5 py-0.5">[{result.citation_id}]</span>
                                        <span>{result.source_file}</span>
                                        {result.section_path && <span>{result.section_path}</span>}
                                        {result.page != null && <span>第 {result.page} 页</span>}
                                      </div>
                                      <div className="mt-2 text-sm text-slate-700 whitespace-pre-wrap leading-6">{result.excerpt || result.content}</div>
                                      <div className="mt-2 text-[11px] text-slate-500 flex flex-wrap gap-3">
                                        <span>score {result.score.toFixed(3)}</span>
                                        <span>relevance {result.relevance.toFixed(3)}</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="py-16 text-center text-sm text-slate-400 space-y-3">
                        <div className="flex justify-center">
                          {selectedSummary ? <Unlink2 className="w-10 h-10 text-slate-300" /> : <Database className="w-10 h-10 text-slate-300" />}
                        </div>
                        <div>{selectedSummary ? '正在等待知识库详情加载。' : '请选择一个知识库查看详情。'}</div>
                      </div>
                    )}
                  </div>
                </section>
              </div>
            </div>
          </div>
        </div>
      </div>

      <ConfirmDialog
        isOpen={showDeleteConfirm}
        title="删除知识库"
        message={`确定删除知识库“${currentKnowledgeBase?.name || selectedSummary?.name || ''}”吗？该操作会移除其向量数据和版本记录。`}
        confirmText="删除"
        cancelText="取消"
        onConfirm={handleDelete}
        onCancel={() => setShowDeleteConfirm(false)}
        variant="danger"
      />
    </>
  );
}
