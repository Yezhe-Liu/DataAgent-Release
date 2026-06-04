import { useState, useEffect, useRef } from 'react';
import { Database, Menu, Wrench, Send } from 'lucide-react';
import { Toaster, toast } from 'sonner';
import { LeftSidebar } from './components/LeftSidebar';
import { KnowledgeBaseModal } from './components/KnowledgeBaseModal';
import { RightSidebar } from './components/RightSidebar';
import { WelcomeScreen } from './components/WelcomeScreen';
import { MessageBubble } from './components/MessageBubble';
import { LoadingIndicator } from './components/LoadingIndicator';
import { WishResultModal } from './components/WishResultModal';
import { ConfigModal } from './components/ConfigModal';
import { AddToolModal } from './components/AddToolModal';
import { GlobalSettingsModal } from './components/GlobalSettingsModal';
import { ConfirmDialog } from './components/ConfirmDialog';
import { Conversation, Message, MCPTool, WishAnalysisResult, MessageChunk, TextBlock, ToolCallBlock, WorkflowTraceBlock, IntentBlock, PlanBlock, StepBlock, RetrievalBlock } from './types';
import { storage } from './utils/storage';
import { sendChatStream, SSEEvent } from './utils/sseClient';
import * as mcpApi from './utils/mcpApi';
import * as chatApi from './utils/chatApi';
import * as ragApi from './utils/ragApi';
import { convertBackendToolToMCPTool, convertMCPToolToBackendConfig, convertRecommendedToolToSuggested } from './utils/mcpHelpers';
import { KnowledgeBaseDetail, KnowledgeBaseSummary, RagRecallResponse } from './utils/ragApi';

type MessageBlock = TextBlock | ToolCallBlock | WorkflowTraceBlock | IntentBlock | PlanBlock | StepBlock | RetrievalBlock;

// Unique ID generator with counter to avoid collisions
let idCounter = 0;
const generateUniqueId = (prefix: string = '') => {
  idCounter++;
  return `${prefix}${Date.now()}-${idCounter}-${Math.random().toString(36).substr(2, 9)}`;
};

const normalizeTimestamp = (value: unknown, fallback: number = Date.now()) => {
  const numericValue = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return fallback;
  }
  return numericValue < 1_000_000_000_000 ? numericValue * 1000 : numericValue;
};

const settleWorkflowBlocks = (blocks: MessageBlock[], stages?: string[]) => {
  const normalizedStages = stages && stages.length > 0
    ? new Set(stages.map((stage) => stage.toLowerCase()))
    : null;

  return blocks.map((block) => {
    if (block.type !== 'workflow' || block.status !== 'running') {
      return block;
    }

    const workflowBlock = block as WorkflowTraceBlock;
    const normalizedStage = workflowBlock.stage.toLowerCase();
    if (normalizedStages && !normalizedStages.has(normalizedStage)) {
      return block;
    }

    return {
      ...workflowBlock,
      status: 'completed' as const,
    };
  });
};

const settleHistoricalMessageBlocks = (blocks?: MessageBlock[]) => {
  if (!blocks) {
    return blocks;
  }
  return settleWorkflowBlocks(blocks);
};

// Migrate old data to ensure unique IDs
const migrateConversationData = (conversations: Conversation[]): Conversation[] => {
  return conversations.map(conv => ({
    ...conv,
    settings: {
      activeKbId: conv.settings?.activeKbId || '',
      ragEnabled: conv.settings?.ragEnabled !== false,
      topKOverride: conv.settings?.topKOverride ?? null,
    },
    createdAt: normalizeTimestamp(conv.createdAt),
    updatedAt: normalizeTimestamp(conv.updatedAt, normalizeTimestamp(conv.createdAt)),
    messages: conv.messages.map(msg => ({
      ...msg,
      id: msg.id || generateUniqueId('msg-'),
      timestamp: normalizeTimestamp(msg.timestamp),
      chunks: msg.chunks?.map(chunk => ({
        ...chunk,
        id: chunk.id || generateUniqueId('chunk-'),
        timestamp: normalizeTimestamp(chunk.timestamp)
      })),
      blocks: settleHistoricalMessageBlocks(msg.blocks?.map(block => ({
        ...block,
        id: block.id || generateUniqueId('block-'),
        timestamp: normalizeTimestamp(block.timestamp)
      })))
    }))
  }));
};

// Default Tavily Tool
const DEFAULT_TAVILY_TOOL: MCPTool = {
  id: 'tavily-search',
  name: 'Tavily 搜索',
  description: '强大的网络搜索工具，可以实时获取最新信息',
  icon: '🔍',
  iconBg: 'bg-blue-50 text-blue-500',
  introduction: 'Tavily 是一个专为 AI 应用优化的搜索引擎，能够快速准确地获取网络信息。',
  config: {
    mcpServers: {
      'tavily-search': {
        command: 'npx',
        args: ['-y', '@tavily/mcp-server'],
        env: {
          TAVILY_API_KEY: 'your-api-key-here'
        }
      }
    }
  },
  enabled: true,
  version: 'v1.0.0',
  author: 'Tavily'
};

export default function App() {
  // UI State
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(false);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(false);
  const [rightSidebarWidth, setRightSidebarWidth] = useState(384);
  const [isLoading, setIsLoading] = useState(false);

  // Modal State
  const [wishResultModal, setWishResultModal] = useState<{
    isOpen: boolean;
    result: WishAnalysisResult | null;
  }>({ isOpen: false, result: null });
  const [configModal, setConfigModal] = useState<{
    isOpen: boolean;
    tool: MCPTool | null;
  }>({ isOpen: false, tool: null });
  const [isKnowledgeBaseModalOpen, setIsKnowledgeBaseModalOpen] = useState(false);
  const [isGlobalSettingsOpen, setIsGlobalSettingsOpen] = useState(false);
  const [addToolModal, setAddToolModal] = useState<{
    isOpen: boolean;
    tool: MCPTool | null;
  }>({ isOpen: false, tool: null });
  const [deleteConfirm, setDeleteConfirm] = useState<{
    isOpen: boolean;
    toolId: string | null;
    toolName: string | null;
  }>({ isOpen: false, toolId: null, toolName: null });

  // Data State
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [mcpTools, setMcpTools] = useState<MCPTool[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseSummary[]>([]);
  const [currentKnowledgeBaseId, setCurrentKnowledgeBaseId] = useState('');
  const [currentKnowledgeBase, setCurrentKnowledgeBase] = useState<KnowledgeBaseDetail | null>(null);
  const [isKnowledgeBasesLoading, setIsKnowledgeBasesLoading] = useState(false);
  const [isKnowledgeBaseDetailLoading, setIsKnowledgeBaseDetailLoading] = useState(false);
  const [isSessionSettingsSaving, setIsSessionSettingsSaving] = useState(false);
  const [isKnowledgeBaseCreating, setIsKnowledgeBaseCreating] = useState(false);
  const [isKnowledgeBaseDeleting, setIsKnowledgeBaseDeleting] = useState(false);
  const [userInput, setUserInput] = useState('');

  // Refs
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const upsertConversation = (conversation: Conversation) => {
    setConversations((prevConvs) => {
      const existing = prevConvs.find((conv) => conv.id === conversation.id);
      if (!existing) {
        return [conversation, ...prevConvs];
      }

      const shouldPreserveLocalMessages = existing.messages.length > 0 && conversation.messages.length === 0;

      return prevConvs.map((conv) => (
        conv.id === conversation.id
          ? {
              ...conv,
              ...conversation,
              messages: shouldPreserveLocalMessages ? existing.messages : conversation.messages,
            }
          : conv
      ));
    });
  };

  const applySessionSettingsToConversation = (sessionId: string, payload: {
    active_kb_id?: string;
    rag_enabled?: boolean;
    top_k_override?: number | null;
  }) => {
    setConversations((prevConvs) => prevConvs.map((conv) => (
      conv.id === sessionId
        ? {
            ...conv,
            settings: {
              activeKbId: String(payload.active_kb_id || ''),
              ragEnabled: payload.rag_enabled !== false,
              topKOverride: payload.top_k_override ?? null,
            },
          }
        : conv
    )));
  };

  const loadSessionSettings = async (conversationId: string) => {
    try {
      const payload = await chatApi.getSessionSettings(conversationId);
      applySessionSettingsToConversation(conversationId, payload);
    } catch (error) {
      console.warn('⚠️ 加载会话设置失败，继续使用本地状态:', error);
    }
  };

  const loadKnowledgeBases = async (preferredKbId?: string) => {
    setIsKnowledgeBasesLoading(true);
    try {
      const list = await ragApi.listKnowledgeBases();
      setKnowledgeBases(list);
      setCurrentKnowledgeBaseId((previous) => {
        const candidate = preferredKbId || previous || currentConversation?.settings?.activeKbId || '';
        if (candidate && list.some((item) => item.id === candidate)) {
          return candidate;
        }
        return list[0]?.id || '';
      });
    } catch (error) {
      toast.error('加载知识库失败', {
        description: error instanceof Error ? error.message : '无法获取知识库列表',
        duration: 4000,
      });
    } finally {
      setIsKnowledgeBasesLoading(false);
    }
  };

  const loadConversationHistory = async (conversationId: string) => {
    try {
      const payload = await chatApi.getHistory(conversationId);
      const conversation = chatApi.convertHistoryToConversation(payload);
      upsertConversation(conversation);
    } catch (error) {
      console.warn('⚠️ 加载会话历史失败，继续使用本地缓存:', error);
    }
  };

  // Load data from localStorage on mount
  useEffect(() => {
    const initializeApp = async () => {
      const savedConversations = storage.getConversations();
      const savedTools = storage.getMCPTools();
      const savedActiveId = storage.getActiveConversationId();

      if (savedTools.length > 0) {
        const hasDefault = savedTools.some(t => t.id === 'tavily-search');
        setMcpTools(hasDefault ? savedTools : [DEFAULT_TAVILY_TOOL, ...savedTools]);
      }

      try {
        const sessions = await chatApi.getSessions();
        const serverConversations = sessions.map(chatApi.convertSessionItemToConversation);
        if (serverConversations.length > 0) {
          setConversations(serverConversations);
          const nextActiveId = savedActiveId && serverConversations.some((conv) => conv.id === savedActiveId)
            ? savedActiveId
            : serverConversations[0].id;
          setActiveConversationId(nextActiveId);
          return;
        }
      } catch (error) {
        console.warn('⚠️ 后端会话列表加载失败，回退到本地缓存:', error);
      }

      if (savedConversations.length > 0) {
        setConversations(migrateConversationData(savedConversations));
        if (savedActiveId) {
          setActiveConversationId(savedActiveId);
        }
      }
    };

    initializeApp();
  }, []);

  useEffect(() => {
    loadMCPTools();
  }, []);

  useEffect(() => {
    if (!activeConversationId) {
      return;
    }

    const activeConversation = conversations.find((conv) => conv.id === activeConversationId);
    if (!activeConversation || activeConversation.messages.length === 0) {
      loadConversationHistory(activeConversationId);
    }
    loadSessionSettings(activeConversationId);
  }, [activeConversationId]);

  useEffect(() => {
    storage.saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    storage.saveMCPTools(mcpTools);
  }, [mcpTools]);

  useEffect(() => {
    storage.setActiveConversationId(activeConversationId);
  }, [activeConversationId]);

  const currentConversation = conversations.find((conversation) => conversation.id === activeConversationId);
  const activeKnowledgeBaseSummary = knowledgeBases.find(
    (item) => item.id === currentConversation?.settings?.activeKbId
  );

  useEffect(() => {
    if (!isKnowledgeBaseModalOpen) {
      return;
    }
    loadKnowledgeBases(currentConversation?.settings?.activeKbId || currentKnowledgeBaseId || '');
  }, [isKnowledgeBaseModalOpen, activeConversationId, currentConversation?.settings?.activeKbId]);

  useEffect(() => {
    if (!isKnowledgeBaseModalOpen) {
      return;
    }
    if (!currentKnowledgeBaseId) {
      setCurrentKnowledgeBase(null);
      return;
    }

    let aborted = false;

    const loadDetail = async () => {
      setIsKnowledgeBaseDetailLoading(true);
      try {
        const detail = await ragApi.getKnowledgeBase(currentKnowledgeBaseId);
        if (!aborted) {
          setCurrentKnowledgeBase(detail);
        }
      } catch (error) {
        if (!aborted) {
          setCurrentKnowledgeBase(null);
          toast.error('加载知识库详情失败', {
            description: error instanceof Error ? error.message : '无法读取知识库详情',
            duration: 4000,
          });
        }
      } finally {
        if (!aborted) {
          setIsKnowledgeBaseDetailLoading(false);
        }
      }
    };

    loadDetail();
    return () => {
      aborted = true;
    };
  }, [isKnowledgeBaseModalOpen, currentKnowledgeBaseId]);

  const handleNewChat = async () => {
    let newConv: Conversation;

    try {
      const created = await chatApi.createSession();
      newConv = chatApi.convertSessionItemToConversation(created);
    } catch (error) {
      console.warn('⚠️ 后端创建会话失败，使用本地兜底:', error);
      newConv = {
        id: generateUniqueId('conv-'),
        title: '新对话',
        messages: [],
        settings: {
          activeKbId: '',
          ragEnabled: true,
          topKOverride: null,
        },
        createdAt: Date.now(),
        updatedAt: Date.now()
      };
    }

    setConversations((prevConvs) => [newConv, ...prevConvs.filter((conv) => conv.id !== newConv.id)]);
    setActiveConversationId(newConv.id);

    setMcpTools((tools) =>
      tools.map((tool) => ({
        ...tool,
        enabled: tool.id === 'tavily-search'
      }))
    );
  };

  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      await chatApi.renameSession(id, newTitle);
    } catch (error) {
      console.warn('⚠️ 后端重命名会话失败，继续更新本地状态:', error);
    }

    setConversations((prevConvs) =>
      prevConvs.map((conv) =>
        conv.id === id
          ? { ...conv, title: newTitle, updatedAt: Date.now() }
          : conv
      )
    );
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await chatApi.deleteSession(id);
    } catch (error) {
      console.warn('⚠️ 后端删除会话失败，继续清理本地状态:', error);
    }

    setConversations((prevConvs) => prevConvs.filter((conv) => conv.id !== id));

    if (id === activeConversationId) {
      const remaining = conversations.filter((conv) => conv.id !== id);
      setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  // 🔥 模块一：从后端加载 MCP 工具列表
  const loadMCPTools = async () => {
    try {
      const backendTools = await mcpApi.getMCPToolList();
      const tools = backendTools.map(convertBackendToolToMCPTool);
      setMcpTools(tools);
      console.log('✅ 已从后端加载工具列表:', tools);
    } catch (error) {
      console.warn('⚠️ 后端加载失败,使用本地存储:', error);
      const savedTools = storage.getMCPTools();
      if (savedTools.length > 0) {
        setMcpTools(savedTools);
      }
    }
  };

  const handleTestConnection = async (
    toolName: string,
    description: string,
    type: string,
    config: any
  ): Promise<{ success: boolean; message: string }> => {
    console.log('🔌 [测试连接] 开始测试:', { toolName, description, type });

    try {
      const result = await mcpApi.testMCPConnection({
        name: toolName,
        description,
        type: type as 'stdio' | 'sse' | 'http',
        config,
      });

      console.log('✅ [测试连接] 后端返回:', result);
      return result;
    } catch (error) {
      console.error('❌ [测试连接] 后端请求失败:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : '连接测试失败',
      };
    }
  };

  const normalizeCustomToolConfig = (toolName: string, rawConfig: any) => {
    let normalizedType: 'stdio' | 'sse' | 'http' = 'stdio';
    let normalizedConfig = rawConfig;

    if (rawConfig && typeof rawConfig === 'object' && rawConfig.mcpServers && typeof rawConfig.mcpServers === 'object') {
      const firstEntry = Object.entries(rawConfig.mcpServers)[0];
      if (!firstEntry) {
        throw new Error('mcpServers 不能为空');
      }

      const [serverName, serverConfig] = firstEntry;
      normalizedConfig = serverConfig;
      if (!toolName && typeof serverName === 'string') {
        toolName = serverName;
      }
    }

    if (normalizedConfig && typeof normalizedConfig === 'object' && 'url' in normalizedConfig) {
      normalizedType = /mcp\.amap\.com\/mcp/i.test(String(normalizedConfig.url)) ? 'http' : 'sse';
    }

    return {
      name: toolName,
      type: normalizedType,
      config: normalizedConfig,
    };
  };

  const handleAddToolTestConnection = async (
    toolName: string,
    description: string,
    configText: string
  ): Promise<{ success: boolean; message: string }> => {
    try {
      const parsedConfig = JSON.parse(configText);
      const normalized = normalizeCustomToolConfig(toolName, parsedConfig);
      return await handleTestConnection(normalized.name, description, normalized.type, normalized.config);
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : '配置格式错误',
      };
    }
  };

  // Send message with SSE streaming
  const handleSendMessageSSE = async (message?: string) => {
    const text = message || userInput.trim();
    if (!text) return;

    // Create conversation if none exists
    let convId = activeConversationId;
    let shouldActivateConversation = false;
    if (!convId) {
      try {
        const created = await chatApi.createSession();
        const newConv = chatApi.convertSessionItemToConversation(created);
        setConversations((prevConvs) => [newConv, ...prevConvs.filter((conv) => conv.id !== newConv.id)]);
        convId = newConv.id;
        shouldActivateConversation = true;
      } catch (error) {
        const newConv: Conversation = {
          id: generateUniqueId('conv-'),
          title: text.substring(0, 30),
          messages: [],
          settings: {
            activeKbId: '',
            ragEnabled: true,
            topKOverride: null,
          },
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setConversations((prevConvs) => [newConv, ...prevConvs]);
        convId = newConv.id;
        shouldActivateConversation = true;
      }
    }

    // Add user message
    const userMessage: Message = {
      id: generateUniqueId('user-msg-'),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    setConversations((prevConvs) =>
      prevConvs.map((conv) =>
        conv.id === convId
          ? {
              ...conv,
              messages: [...conv.messages, userMessage],
              title: conv.messages.length === 0 ? text.substring(0, 30) : conv.title,
              updatedAt: Date.now(),
            }
          : conv
      )
    );

    setUserInput('');

    try {
      // Create initial assistant message with blocks
      const assistantMessageId = generateUniqueId('assistant-msg-');
      const blocks: MessageBlock[] = [];

      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        blocks: blocks,
        timestamp: Date.now(),
        isComplete: false,
        isStreaming: true,
      };

      // Add empty assistant message
      setConversations((prevConvs) =>
        prevConvs.map((conv) =>
          conv.id === convId
            ? {
                ...conv,
                messages: [...conv.messages, assistantMessage],
                updatedAt: Date.now(),
              }
            : conv
        )
      );

      if (shouldActivateConversation) {
        setActiveConversationId(convId);
      }

      const conversationRagSettings = conversations.find((conv) => conv.id === convId)?.settings;

      // Handle SSE events
      await sendChatStream(
        text,
        convId,
        (event: SSEEvent) => {
          setConversations((prevConvs) =>
            prevConvs.map((conv) => {
              if (conv.id !== convId) return conv;

              const updatedMessages = conv.messages.map((msg) => {
                if (msg.id !== assistantMessageId) return msg;

                let newBlocks = [...(msg.blocks || [])] as MessageBlock[];

                switch (event.type) {
                  case 'token': {
                    newBlocks = settleWorkflowBlocks(newBlocks, ['answer']);
                    // Append text to last text block or create new one
                    const lastBlock = newBlocks[newBlocks.length - 1];
                    if (lastBlock && lastBlock.type === 'text') {
                      // Append to existing text block
                      (lastBlock as TextBlock).content += event.data.content;
                    } else {
                      // Create new text block
                      const textBlock: TextBlock = {
                        id: generateUniqueId('text-'),
                        type: 'text',
                        content: event.data.content,
                        timestamp: Date.now(),
                      };
                      newBlocks.push(textBlock);
                    }
                    break;
                  }

                  case 'tool_start': {
                    // Create tool block
                    const toolBlock: ToolCallBlock = {
                      id: generateUniqueId('tool-'),
                      type: 'tool_call',
                      toolName: event.data.tool_name,
                      input: event.data.input,
                      status: 'running',
                      timestamp: Date.now(),
                      isExpanded: false,
                    };
                    newBlocks.push(toolBlock);
                    break;
                  }

                  case 'tool_end': {
                    // Find the most recent running tool block with matching name
                    for (let i = newBlocks.length - 1; i >= 0; i--) {
                      const block = newBlocks[i];
                      if (
                        block.type === 'tool_call' &&
                        (block as ToolCallBlock).toolName === event.data.tool_name &&
                        (block as ToolCallBlock).status === 'running'
                      ) {
                        (block as ToolCallBlock).output = event.data.output;
                        (block as ToolCallBlock).status = 'completed';
                        break;
                      }
                    }
                    break;
                  }

                  case 'tool_error': {
                    for (let i = newBlocks.length - 1; i >= 0; i--) {
                      const block = newBlocks[i];
                      if (
                        block.type === 'tool_call' &&
                        (block as ToolCallBlock).toolName === event.data.tool_name &&
                        (block as ToolCallBlock).status === 'running'
                      ) {
                        (block as ToolCallBlock).output = event.data.error;
                        (block as ToolCallBlock).status = 'failed';
                        break;
                      }
                    }
                    break;
                  }

                  case 'workflow': {
                    if (event.data.status !== 'running') {
                      newBlocks = settleWorkflowBlocks(newBlocks, [event.data.stage || 'workflow']);
                    }
                    const workflowBlock: WorkflowTraceBlock = {
                      id: generateUniqueId('workflow-'),
                      type: 'workflow',
                      stage: event.data.stage || 'workflow',
                      message: event.data.message || '工作流执行中',
                      status: event.data.status === 'running' ? 'running' : 'completed',
                      route: event.data.route,
                      timestamp: Date.now(),
                    };
                    newBlocks.push(workflowBlock);
                    break;
                  }

                  case 'intent': {
                    newBlocks = settleWorkflowBlocks(newBlocks, ['intent']);
                    const intentBlock: IntentBlock = {
                      id: generateUniqueId('intent-'),
                      type: 'intent',
                      mode: event.data.mode || 'simple_qa',
                      reason: event.data.reason || '',
                      candidateTools: event.data.candidate_tools || [],
                      timestamp: Date.now(),
                    };
                    newBlocks.push(intentBlock);
                    break;
                  }

                  case 'plan': {
                    newBlocks = settleWorkflowBlocks(newBlocks, ['plan', 'planning']);
                    const steps = (event.data.steps || []).map((s: any) => ({
                      id: s.id || '',
                      tool: s.tool || '',
                      goal: s.goal || '',
                    }));
                    const planBlock: PlanBlock = {
                      id: generateUniqueId('plan-'),
                      type: 'plan',
                      rationale: event.data.rationale || '',
                      steps,
                      timestamp: Date.now(),
                    };
                    newBlocks.push(planBlock);
                    break;
                  }

                  case 'step_start': {
                    const stepBlock: StepBlock = {
                      id: generateUniqueId('step-'),
                      type: 'step',
                      stepId: event.data.step_id || '',
                      tool: event.data.tool || '',
                      goal: event.data.goal || '',
                      args: event.data.args,
                      status: 'running',
                      timestamp: Date.now(),
                    };
                    newBlocks.push(stepBlock);
                    break;
                  }

                  case 'step_end': {
                    for (let i = newBlocks.length - 1; i >= 0; i--) {
                      const block = newBlocks[i];
                      if (
                        block.type === 'step' &&
                        (block as StepBlock).stepId === event.data.step_id &&
                        (block as StepBlock).status === 'running'
                      ) {
                        (block as StepBlock).output = event.data.output;
                        (block as StepBlock).durationMs = event.data.duration_ms;
                        (block as StepBlock).status = 'completed';
                        break;
                      }
                    }
                    break;
                  }

                  case 'step_error': {
                    for (let i = newBlocks.length - 1; i >= 0; i--) {
                      const block = newBlocks[i];
                      if (
                        block.type === 'step' &&
                        (block as StepBlock).stepId === event.data.step_id &&
                        (block as StepBlock).status === 'running'
                      ) {
                        (block as StepBlock).error = event.data.error;
                        (block as StepBlock).status = 'failed';
                        break;
                      }
                    }
                    break;
                  }

                  case 'retrieval': {
                    newBlocks = settleWorkflowBlocks(newBlocks, ['retrieval']);
                    const docs = Array.isArray(event.data?.documents) ? event.data.documents : [];
                    if (docs.length === 0) {
                      break; // No hits: avoid rendering an empty card
                    }
                    const retrievalBlock: RetrievalBlock = {
                      id: generateUniqueId('retrieval-'),
                      type: 'retrieval',
                      query: event.data?.query || '',
                      strategy: event.data?.strategy,
                      documents: docs.map((d: any) => ({
                        id: d.id || '',
                        title: d.title || '',
                        category: d.category || '',
                        score: typeof d.score === 'number' ? d.score : Number(d.score) || 0,
                        keyword_score: typeof d.keyword_score === 'number' ? d.keyword_score : undefined,
                        vector_score: typeof d.vector_score === 'number' ? d.vector_score : undefined,
                        relevance: typeof d.relevance === 'number' ? d.relevance : undefined,
                        excerpt: d.excerpt,
                        content: d.content,
                        source_file: d.source_file,
                        section_path: d.section_path,
                        page: typeof d.page === 'number' ? d.page : undefined,
                        citation_id: typeof d.citation_id === 'number' ? d.citation_id : undefined,
                        metadata: d.metadata,
                      })),
                      timestamp: Date.now(),
                    };
                    newBlocks.push(retrievalBlock);
                    break;
                  }

                  case 'finish': {
                    newBlocks = settleWorkflowBlocks(newBlocks);
                    // Mark message as complete
                    return { ...msg, blocks: newBlocks, isComplete: true, isStreaming: false };
                  }

                  case 'error': {
                    newBlocks = settleWorkflowBlocks(newBlocks);
                    // Handle error from backend
                    for (let i = newBlocks.length - 1; i >= 0; i--) {
                      const block = newBlocks[i];
                      if (block.type === 'tool_call' && (block as ToolCallBlock).status === 'running') {
                        (block as ToolCallBlock).output = event.data.message || '后端处理出错';
                        (block as ToolCallBlock).status = 'failed';
                      }
                    }
                    console.error('Backend error:', event.data.message);
                    toast.error('后端错误', {
                      description: event.data.message || '后端处理出错',
                      duration: 5000,
                    });
                    return { ...msg, blocks: newBlocks, isComplete: true, isStreaming: false };
                  }
                }

                return { ...msg, blocks: newBlocks };
              });

              return { ...conv, messages: updatedMessages, updatedAt: Date.now() };
            })
          );
        },
        (error: Error) => {
          console.error('SSE error:', error);
          toast.error('连接失败', {
            description: '无法连接到后端服务，请检查服务是否启动',
            duration: 5000,
          });
          setIsLoading(false);
        },
        {
          kbId: conversationRagSettings?.activeKbId,
          ragEnabled: conversationRagSettings?.ragEnabled,
          topKOverride: conversationRagSettings?.topKOverride,
        }
      );

      // Mark as complete
      setConversations((prevConvs) =>
        prevConvs.map((conv) =>
          conv.id === convId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, isComplete: true, isStreaming: false }
                    : msg
                ),
                updatedAt: Date.now(),
              }
            : conv
        )
      );
    } catch (error) {
      console.error('Error generating response:', error);
      toast.error('发送失败', {
        description: '消息发送失败,请重试',
        duration: 4000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Send message with SSE streaming
  const handleSendMessage = async (message?: string) => {
    const text = message || userInput.trim();
    if (!text) return;

    setIsLoading(true);

    try {
      await handleSendMessageSSE(text);
    } catch (error) {
      console.error('SSE failed:', error);
      toast.error('发送消息失败', {
        description: error instanceof Error ? error.message : '无法连接到服务器',
        duration: 4000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle wish processing
  const handleProcessWish = async (wish: string) => {
    try {
      const recommendedTools = await mcpApi.searchMCPToolsAI(wish);
      const suggestedTools = recommendedTools.map(convertRecommendedToolToSuggested);
      setWishResultModal({
        isOpen: true,
        result: { wish, suggestedTools },
      });
    } catch (error) {
      console.error('AI search failed:', error);
      toast.error('智能分析失败', {
        description: error instanceof Error ? error.message : '无法连接到服务器',
        duration: 4000,
      });
    }
  };

  // Confirm adding tools from wish
  const handleConfirmAddTools = async (selectedIndices: number[]) => {
    if (!wishResultModal.result || selectedIndices.length === 0) return;

    try {
      const toolsToInstall = selectedIndices.map((index) => {
        const tool = wishResultModal.result!.suggestedTools[index];
        return {
          name: tool.name,
          description: tool.description,
          type: (tool.type || 'stdio') as 'stdio' | 'sse' | 'http',
          config: tool.defaultConfig || tool.config,
        };
      });

      // 批量安装工具
      await mcpApi.batchInstallMCPTools(toolsToInstall);

      // 🔥 后端默认是激活状态，我们需要手动设置为非激活
      // 并发调用 toggle 接口将所有工具设为 active: false
      const togglePromises = toolsToInstall.map((tool) =>
        mcpApi.toggleMCPTool(tool.name, false).catch((err) => {
          console.warn(`Failed to disable tool ${tool.name}:`, err);
        })
      );
      await Promise.all(togglePromises);

      // 重新加载工具列表
      await loadMCPTools();

      setWishResultModal({ isOpen: false, result: null });

      const toolNames = toolsToInstall.map((t) => t.name).join('、');
      toast.success(`已新增 ${toolNames}`, {
        description: '工具已添加，默认未激活。请点击工具名称配置后再激活！',
        duration: 4000,
      });
    } catch (error) {
      toast.error('批量添加失败', {
        description: error instanceof Error ? error.message : '无法添加工具',
        duration: 4000,
      });
    }
  };

  // Toggle tool enabled state (with connection test when enabling)
  const handleToggleTool = async (toolId: string) => {
    const tool = mcpTools.find((t) => t.id === toolId);
    if (!tool) return;

    if (!tool.enabled) {
      const toastId = toast.loading(`正在测试 ${tool.name} 连接...`);

      try {
        const backendConfig = convertMCPToolToBackendConfig(tool);
        const result = await mcpApi.testMCPConnection(backendConfig);

        if (result.success) {
          await mcpApi.toggleMCPTool(tool.name, true);

          toast.success(`${tool.name} 连接测试成功`, {
            id: toastId,
            description: result.message,
            duration: 3000,
          });

          setMcpTools((tools) =>
            tools.map((t) => t.id === toolId ? { ...t, enabled: true } : t)
          );
        } else {
          toast.error(`${tool.name} 连接测试失败`, {
            id: toastId,
            description: result.message,
            duration: 4000,
          });
        }
      } catch (error) {
        toast.error(`${tool.name} 连接测试失败`, {
          id: toastId,
          description: error instanceof Error ? error.message : '请检查配置后重试',
          duration: 4000,
        });
      }
    } else {
      try {
        await mcpApi.toggleMCPTool(tool.name, false);
        setMcpTools((tools) =>
          tools.map((t) => t.id === toolId ? { ...t, enabled: false } : t)
        );
        toast.info(`已停用 ${tool.name}`, { duration: 2000 });
      } catch (error) {
        toast.error('操作失败', {
          description: error instanceof Error ? error.message : '无法停用工具',
          duration: 3000,
        });
      }
    }
  };

  // Open config modal
  const handleOpenConfig = (toolId: string) => {
    const tool = mcpTools.find((t) => t.id === toolId);
    if (tool) {
      setConfigModal({ isOpen: true, tool });
    }
  };

  // Save tool config
  const handleSaveConfig = async (toolId: string, description: string, config: string) => {
    try {
      const parsedConfig = JSON.parse(config);
      const tool = mcpTools.find((t) => t.id === toolId);
      if (!tool) return;

      await mcpApi.installMCPTool({
        name: tool.name,
        description,
        type: tool.type || 'stdio',
        config: parsedConfig,
      });

      setMcpTools((tools) =>
        tools.map((t) =>
          t.id === toolId ? { ...t, description, config: parsedConfig } : t
        )
      );

      toast.success('配置已保存', {
        description: '工具配置更新成功',
        duration: 3000,
      });
    } catch (error) {
      console.error('Save config failed:', error);
      toast.error('保存失败', {
        description: error instanceof Error ? error.message : '配置格式错误',
        duration: 4000,
      });
      throw error;
    }
  };

  // Delete tool
  const handleDeleteTool = (toolId: string) => {
    const tool = mcpTools.find((t) => t.id === toolId);
    if (!tool) return;

    // Close config modal first
    setConfigModal({ isOpen: false, tool: null });

    // Show confirmation
    setDeleteConfirm({ isOpen: true, toolId, toolName: tool.name });
  };

  // Confirm delete tool
  const handleConfirmDeleteTool = async () => {
    const { toolId, toolName } = deleteConfirm;
    if (!toolId || !toolName) return;

    try {
      await mcpApi.deleteMCPTool(toolName);
      setMcpTools((tools) => tools.filter((t) => t.id !== toolId));

      toast.success(`已删除工具"${toolName}"`, { duration: 3000 });
    } catch (error) {
      toast.error('删除失败', {
        description: error instanceof Error ? error.message : '无法删除工具',
        duration: 4000,
      });
    } finally {
      setDeleteConfirm({ isOpen: false, toolId: null, toolName: null });
    }
  };

  // Add new tool manually
  const handleAddTool = async (name: string, description: string, introduction: string, config: string) => {
    try {
      const parsedConfig = JSON.parse(config);
      const normalized = normalizeCustomToolConfig(name, parsedConfig);

      await mcpApi.installMCPTool({
        name: normalized.name,
        description,
        type: normalized.type,
        config: normalized.config,
      });

      await loadMCPTools();

      toast.success(`已添加工具"${normalized.name}"`, {
        description: '工具已成功保存到后端配置',
        duration: 3000,
      });
    } catch (error) {
      toast.error('配置保存失败', {
        description: error instanceof Error ? error.message : '请检查 JSON 配置格式是否正确',
        duration: 4000,
      });
      throw error;
    }
  };

  const handleSaveKnowledgeBaseSessionSettings = async (payload: {
    activeKbId: string;
    ragEnabled: boolean;
    topKOverride: number | null;
  }) => {
    if (!activeConversationId) {
      toast.warning('请先选择或创建一个会话', {
        duration: 3000,
      });
      return;
    }

    setIsSessionSettingsSaving(true);
    try {
      const response = await chatApi.updateSessionSettings(activeConversationId, {
        active_kb_id: payload.activeKbId,
        rag_enabled: payload.ragEnabled,
        top_k_override: payload.topKOverride,
      });
      applySessionSettingsToConversation(activeConversationId, response);
      toast.success('会话知识库设置已保存', {
        duration: 3000,
      });
    } catch (error) {
      toast.error('保存会话设置失败', {
        description: error instanceof Error ? error.message : '无法更新当前会话的知识库设置',
        duration: 4000,
      });
      throw error;
    } finally {
      setIsSessionSettingsSaving(false);
    }
  };

  const handleCreateKnowledgeBase = async (payload: {
    name: string;
    description: string;
    files: File[];
    chunkSize?: number;
    chunkOverlap?: number;
    topK?: number;
  }) => {
    if (!payload.name.trim()) {
      toast.warning('请输入知识库名称');
      return;
    }
    if (payload.files.length === 0) {
      toast.warning('请至少选择一个文档文件');
      return;
    }

    setIsKnowledgeBaseCreating(true);
    try {
      const uploadResponse = await ragApi.uploadRagFiles(payload.files);
      const createResponse = await ragApi.createKnowledgeBase({
        name: payload.name.trim(),
        description: payload.description.trim(),
        upload_ids: uploadResponse.files.map((item) => item.upload_id),
        chunk_size: payload.chunkSize,
        chunk_overlap: payload.chunkOverlap,
        top_k: payload.topK,
      });

      setCurrentKnowledgeBase(createResponse.kb);
      setCurrentKnowledgeBaseId(createResponse.kb.id);
      await loadKnowledgeBases(createResponse.kb.id);

      toast.success('知识库创建成功', {
        description: `已生成 ${createResponse.total_chunks} 个文档片段`,
        duration: 4000,
      });
    } catch (error) {
      toast.error('创建知识库失败', {
        description: error instanceof Error ? error.message : '无法完成文档上传和建库',
        duration: 4000,
      });
      throw error;
    } finally {
      setIsKnowledgeBaseCreating(false);
    }
  };

  const handleDeleteKnowledgeBase = async (kbId: string) => {
    setIsKnowledgeBaseDeleting(true);
    try {
      await ragApi.deleteKnowledgeBase(kbId);

      if (activeConversationId && currentConversation?.settings?.activeKbId === kbId) {
        const response = await chatApi.updateSessionSettings(activeConversationId, {
          active_kb_id: '',
        });
        applySessionSettingsToConversation(activeConversationId, response);
      }

      setCurrentKnowledgeBase(null);
      setCurrentKnowledgeBaseId('');
      await loadKnowledgeBases('');

      toast.success('知识库已删除', {
        duration: 3000,
      });
    } catch (error) {
      toast.error('删除知识库失败', {
        description: error instanceof Error ? error.message : '无法删除知识库',
        duration: 4000,
      });
      throw error;
    } finally {
      setIsKnowledgeBaseDeleting(false);
    }
  };

  const handleRecallKnowledgeBase = async (kbId: string, query: string, topK?: number): Promise<RagRecallResponse> => {
    return ragApi.recallKnowledgeBase(kbId, query, topK);
  };

  // Handle Enter key in textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="h-screen flex text-slate-800 overflow-hidden relative">
      {/* Left Sidebar */}
      <LeftSidebar
        isOpen={isLeftSidebarOpen}
        onClose={() => setIsLeftSidebarOpen(false)}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewChat={handleNewChat}
        onSelectConversation={setActiveConversationId}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main Content - Dynamic width based on panels */}
      <div
        className="fixed top-0 bottom-0 flex flex-col bg-slate-50/50 transition-all duration-300 z-10"
        style={{
          left: '0',
          right: isRightSidebarOpen ? `${rightSidebarWidth}px` : '0',
        }}
      >
        {/* Top Navigation */}
        <div className="h-16 border-b border-slate-200 bg-white flex items-center justify-between px-4 shadow-sm z-10">
          {/* Left Section: Menu */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsLeftSidebarOpen(true)}
              className="p-2 text-slate-500 hover:text-indigo-600 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
            >
              <Menu className="w-5 h-5" />
            </button>
          </div>

          {/* Center Section: Title + Gitee link */}
          <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center space-x-3">
            <h1 className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent text-2xl tracking-tight">
              MCPChat
            </h1>
            {activeConversationId && (
              <div className="hidden xl:flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                <span className={`inline-block h-2 w-2 rounded-full ${currentConversation?.settings?.ragEnabled !== false ? 'bg-emerald-500' : 'bg-slate-300'}`}></span>
                <span>{currentConversation?.settings?.ragEnabled !== false ? 'RAG 已启用' : 'RAG 已关闭'}</span>
                <span className="text-slate-300">|</span>
                <span className="truncate max-w-[220px]">
                  {activeKnowledgeBaseSummary?.name || currentConversation?.settings?.activeKbId || '未绑定知识库'}
                </span>
              </div>
            )}
            <a
              href="https://gitee.com/ye_sheng0839/mcp-agent"
              target="_blank"
              rel="noopener noreferrer"
              className={`transition-all cursor-pointer hover:opacity-80 ${
                isRightSidebarOpen ? 'hidden' : 'hidden lg:block'
              }`}
            >
              <img src="/gitee.svg" alt="Gitee" className="w-6 h-6" />
            </a>
          </div>

          {/* Right Section: MCP Toolbox */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsKnowledgeBaseModalOpen(true)}
              className="flex items-center justify-center space-x-2 px-3 py-1.5 rounded-lg transition-colors border text-sm cursor-pointer bg-emerald-50 text-emerald-700 border-emerald-100 hover:bg-emerald-100"
            >
              <Database className="w-4 h-4" />
              <span className="hidden sm:inline">知识库</span>
            </button>
            <button
              onClick={() => setIsRightSidebarOpen(!isRightSidebarOpen)}
              className={`flex items-center justify-center space-x-2 px-3 py-1.5 rounded-lg transition-colors border text-sm cursor-pointer ${
                isRightSidebarOpen
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-indigo-50 text-indigo-600 border-indigo-100 hover:bg-indigo-100'
              }`}
            >
              <Wrench className="w-4 h-4" />
              <span className={isRightSidebarOpen ? 'hidden' : 'hidden sm:inline'}>
                MCP 工具箱
              </span>
            </button>
          </div>
        </div>

        {/* Chat Container */}
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
        >
          {!currentConversation || currentConversation.messages.length === 0 ? (
            <WelcomeScreen onSendMessage={handleSendMessage} />
          ) : (
            <>
              {currentConversation.messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </>
          )}
          {isLoading && <LoadingIndicator />}
        </div>

        {/* Input Area */}
        <div className="p-6 pt-2 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent">
          {activeConversationId && (
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 border ${currentConversation?.settings?.ragEnabled !== false ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-white text-slate-500'}`}>
                <span className={`inline-block h-1.5 w-1.5 rounded-full ${currentConversation?.settings?.ragEnabled !== false ? 'bg-emerald-500' : 'bg-slate-300'}`}></span>
                {currentConversation?.settings?.ragEnabled !== false ? '本会话启用知识库检索' : '本会话已关闭知识库检索'}
              </span>
              <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1">
                <Database className="w-3 h-3" />
                {activeKnowledgeBaseSummary?.name || currentConversation?.settings?.activeKbId || '未绑定知识库'}
              </span>
              {currentConversation?.settings?.topKOverride != null && (
                <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1">
                  Top K {currentConversation.settings.topKOverride}
                </span>
              )}
            </div>
          )}
          <div className="relative bg-white rounded-xl shadow-lg border border-slate-200 transition-all duration-200 focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-500/20">
            <textarea
              ref={textareaRef}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              className="w-full p-4 pr-12 bg-transparent border-none focus:ring-0 focus:outline-none resize-none max-h-48 text-slate-700 placeholder-slate-400 rounded-xl"
              placeholder="输入消息... (例如: 帮我写一个 Python 脚本)"
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={isLoading || !userInput.trim()}
              className="absolute right-2 bottom-2 p-2 w-10 h-10 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors flex items-center justify-center shadow-md z-10 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Right Sidebar - Fixed position with dynamic width */}
      <RightSidebar
        isOpen={isRightSidebarOpen}
        onClose={() => setIsRightSidebarOpen(false)}
        tools={mcpTools}
        onToggleTool={handleToggleTool}
        onOpenConfig={handleOpenConfig}
        onProcessWish={handleProcessWish}
        onOpenGlobalSettings={() => setIsGlobalSettingsOpen(true)}
        onOpenAddTool={() => setAddToolModal({ isOpen: true, tool: null })}
        onWidthChange={setRightSidebarWidth}
      />

      {/* Modals */}
      <WishResultModal
        isOpen={wishResultModal.isOpen}
        onClose={() => setWishResultModal({ isOpen: false, result: null })}
        result={wishResultModal.result}
        onConfirm={handleConfirmAddTools}
      />

      <ConfigModal
        isOpen={configModal.isOpen}
        onClose={() => setConfigModal({ isOpen: false, tool: null })}
        tool={configModal.tool}
        onSave={handleSaveConfig}
        onDelete={handleDeleteTool}
        onTestConnection={handleTestConnection}
      />

      <AddToolModal
        isOpen={addToolModal.isOpen}
        onClose={() => setAddToolModal({ isOpen: false, tool: null })}
        onAdd={handleAddTool}
        onTestConnection={handleAddToolTestConnection}
      />

      <GlobalSettingsModal
        isOpen={isGlobalSettingsOpen}
        onClose={() => setIsGlobalSettingsOpen(false)}
      />

      <KnowledgeBaseModal
        isOpen={isKnowledgeBaseModalOpen}
        onClose={() => setIsKnowledgeBaseModalOpen(false)}
        knowledgeBases={knowledgeBases}
        currentKnowledgeBase={currentKnowledgeBase}
        currentKnowledgeBaseId={currentKnowledgeBaseId}
        activeKbId={currentConversation?.settings?.activeKbId || ''}
        ragEnabled={currentConversation?.settings?.ragEnabled !== false}
        topKOverride={currentConversation?.settings?.topKOverride ?? null}
        loadingList={isKnowledgeBasesLoading}
        loadingDetail={isKnowledgeBaseDetailLoading}
        savingSession={isSessionSettingsSaving}
        creatingKnowledgeBase={isKnowledgeBaseCreating}
        deletingKnowledgeBase={isKnowledgeBaseDeleting}
        onRefreshKnowledgeBases={() => loadKnowledgeBases(currentConversation?.settings?.activeKbId || currentKnowledgeBaseId || '')}
        onSelectKnowledgeBase={setCurrentKnowledgeBaseId}
        onSaveSessionSettings={handleSaveKnowledgeBaseSessionSettings}
        onCreateKnowledgeBase={handleCreateKnowledgeBase}
        onDeleteKnowledgeBase={handleDeleteKnowledgeBase}
        onRecallKnowledgeBase={handleRecallKnowledgeBase}
      />

      {/* Delete Confirmation */}
      <ConfirmDialog
        isOpen={deleteConfirm.isOpen}
        title="删除工具"
        message={`确定要删除工具"${deleteConfirm.toolName}"吗？此操作无法撤销。`}
        onConfirm={handleConfirmDeleteTool}
        onCancel={() => setDeleteConfirm({ isOpen: false, toolId: null, toolName: null })}
        variant="danger"
      />

      {/* Toast Notifications */}
      <Toaster position="top-right" richColors closeButton />
    </div>
  );
}