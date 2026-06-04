// MCP Helper Functions - 后端数据与前端数据转换
import * as mcpApi from './mcpApi';
import { MCPTool } from '../types';

/**
 * 将后端工具数据转换为前端 MCPTool 格式
 */
export function convertBackendToolToMCPTool(backendTool: mcpApi.MCPToolResponse): MCPTool {
  let config;
  try {
    config = JSON.parse(backendTool.config_json);
  } catch (e) {
    config = { mcpServers: {} };
  }

  // 根据类型确定图标
  const isRemoteTool = backendTool.type === 'sse' || backendTool.type === 'http';
  const icon = isRemoteTool ? '🌐' : '⚡';
  const iconBg = isRemoteTool 
    ? 'bg-purple-50 text-purple-500' 
    : 'bg-green-50 text-green-500';

  return {
    id: backendTool.name, // 使用 name 作为 ID
    name: backendTool.name,
    description: backendTool.description,
    icon: icon,
    iconBg: iconBg,
    introduction: backendTool.description,
    config: config,
    enabled: backendTool.active,
    type: backendTool.type,
    version: 'v1.0.0',
    author: 'MCP'
  };
}

/**
 * 将前端 MCPTool 转换为后端配置格式
 */
export function convertMCPToolToBackendConfig(tool: MCPTool): mcpApi.MCPToolConfig {
  return {
    name: tool.name,
    description: tool.description,
    type: (tool.type || 'stdio') as 'stdio' | 'sse' | 'http',
    config: tool.config
  };
}

/**
 * 将推荐工具转换为前端格式
 */
export function convertRecommendedToolToSuggested(recTool: mcpApi.RecommendedTool) {
  const isRemoteTool = recTool.type === 'sse' || recTool.type === 'http';
  const icon = isRemoteTool ? '🌐' : '⚡';
  const iconBg = isRemoteTool 
    ? 'bg-purple-50 text-purple-500' 
    : 'bg-green-50 text-green-500';

  return {
    name: recTool.name,
    description: recTool.description,
    icon: icon,
    iconBg: iconBg,
    functions: [],
    config: recTool.default_config || { mcpServers: {} },
    isNew: !recTool.installed,
    recommendReason: recTool.recommend_reason,
    installed: recTool.installed,
    type: recTool.type,
    defaultConfig: recTool.default_config
  };
}
