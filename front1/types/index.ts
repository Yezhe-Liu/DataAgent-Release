// Type definitions for the application

// 工具调用状态
export interface ToolCallBlock {
  id: string;
  type: 'tool_call';
  toolName: string;
  input: any;
  output?: any;
  status: 'running' | 'completed' | 'failed';
  timestamp: number;
  isExpanded?: boolean; // 是否展开查看详情
}

// 文本块类型
export interface TextBlock {
  id: string;
  type: 'text';
  content: string;
  timestamp: number;
}

// 工作流跟踪块类型
export interface WorkflowTraceBlock {
  id: string;
  type: 'workflow';
  stage: string;
  message: string;
  status: 'running' | 'completed';
  route?: string;
  timestamp: number;
}

// 意图分析块
export interface IntentBlock {
  id: string;
  type: 'intent';
  mode: 'simple_qa' | 'retrieval_qa' | 'tool_task';
  reason: string;
  candidateTools: string[];
  timestamp: number;
}

// 计划块
export interface PlanBlock {
  id: string;
  type: 'plan';
  rationale: string;
  steps: { id: string; tool: string; goal: string }[];
  timestamp: number;
}

// 步骤执行块
export interface StepBlock {
  id: string;
  type: 'step';
  stepId: string;
  tool: string;
  goal: string;
  args?: any;
  output?: string;
  error?: string;
  durationMs?: number;
  status: 'running' | 'completed' | 'failed';
  timestamp: number;
}

// 知识库检索结果块
export interface RetrievalDocument {
  id: string;
  title: string;
  category: string;
  score: number;
  keyword_score?: number;
  vector_score?: number;
  relevance?: number;
  excerpt?: string;
  content?: string;
  source_file?: string;
  section_path?: string;
  page?: number | null;
  citation_id?: number;
  metadata?: Record<string, any>;
}

export interface RetrievalBlock {
  id: string;
  type: 'retrieval';
  query: string;
  strategy?: string;
  documents: RetrievalDocument[];
  timestamp: number;
}

// 消息块类型 - 支持流式响应（保留兼容性）
export interface MessageChunk {
  id: string;
  type: 'thought' | 'tool_call' | 'content';
  content: string;
  timestamp: number;
  isStreaming?: boolean; // 是否正在流式输出
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content?: string; // 用户消息的内容
  chunks?: MessageChunk[]; // AI消息的chunks（旧格式，保留兼容性）
  blocks?: (TextBlock | ToolCallBlock | WorkflowTraceBlock | IntentBlock | PlanBlock | StepBlock | RetrievalBlock)[]; // 新格式：支持意图、计划、步骤、检索等新块类型
  timestamp: number;
  isComplete?: boolean; // 消息是否完成
  isStreaming?: boolean; // 是否正在流式输出（用于显示光标）
}

export interface ThoughtStep {
  type: 'thought' | 'action' | 'observation';
  content: string;
}

export interface SearchResult {
  title: string;
  site: string;
  description: string;
  icon: string;
  url?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  settings?: {
    activeKbId?: string;
    ragEnabled?: boolean;
    topKOverride?: number | null;
  };
  createdAt: number;
  updatedAt: number;
}

export interface MCPTool {
  id: string;
  name: string;
  description: string;
  icon: string;
  iconBg: string;
  introduction: string; // 工具的详细介绍
  config: MCPConfig;
  enabled: boolean;
  version?: string;
  author?: string;
  type?: 'stdio' | 'sse' | 'http'; // 后端返回的类型
}

export interface MCPConfig {
  mcpServers: {
    [key: string]: {
      command: string;
      args: string[];
      env?: Record<string, string>;
    };
  };
}

export interface WishAnalysisResult {
  wish: string;
  suggestedTools: SuggestedTool[];
}

export interface SuggestedTool {
  name: string;
  description: string;
  icon: string;
  iconBg: string;
  functions: string[];
  config: MCPConfig;
  isNew: boolean;
  recommendReason?: string; // AI 推荐理由
  installed?: boolean; // 是否已安装
  type?: 'stdio' | 'sse' | 'http'; // 工具类型
  defaultConfig?: any; // 默认配置
}