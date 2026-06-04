import { X, Copy, Check, Plug, Trash2, Save } from 'lucide-react';
import { toast } from 'sonner';
import { MCPTool } from '../types';
import { useState, useEffect } from 'react';

interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  tool: MCPTool | null;
  onSave: (toolId: string, description: string, config: string) => Promise<void>;
  onDelete?: (toolId: string) => void;
  onTestConnection: (toolName: string, description: string, type: string, config: any) => Promise<{ success: boolean; message: string }>;
}

export function ConfigModal({ isOpen, onClose, tool, onSave, onDelete, onTestConnection }: ConfigModalProps) {
  const [config, setConfig] = useState('');
  const [description, setDescription] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isTested, setIsTested] = useState(false);

  // 当 tool 改变时重置状态
  useEffect(() => {
    if (tool) {
      setConfig(JSON.stringify(tool.config, null, 2));
      setDescription(tool.description || '');
      setIsTested(false);
      setIsSaved(false);
    }
  }, [tool]);

  if (!isOpen || !tool) return null;

  const handleTestConnection = async () => {
    setIsTesting(true);
    
    try {
      // Parse config
      const parsedConfig = JSON.parse(config);
      
      // Call test connection
      const result = await onTestConnection(
        tool.name,
        description,
        tool.type || 'stdio',
        parsedConfig
      );
      
      setIsTesting(false);
      
      if (result.success) {
        setIsTested(true);
        toast.success('连接测试成功', {
          description: result.message,
          duration: 3000,
        });
      } else {
        toast.error('连接测试失败', {
          description: result.message,
          duration: 4000,
        });
      }
    } catch (error) {
      setIsTesting(false);
      toast.error('配置格式错误', {
        description: error instanceof Error ? error.message : '请检查 JSON 格式',
        duration: 4000,
      });
    }
  };

  const handleSave = async () => {
    if (!isTested) {
      toast.warning('请先测试连接', {
        description: '确保配置正确后再保存',
        duration: 3000,
      });
      return;
    }

    setIsSaving(true);
    try {
      await onSave(tool.id, description, config);
      setIsSaving(false); // 🔥 成功后立即重置 isSaving
      setIsSaved(true);
      setTimeout(() => {
        setIsSaved(false);
        onClose();
      }, 800);
    } catch (error) {
      setIsSaving(false);
    }
  };

  const getIconComponent = (iconName: string) => {
    return <span className="text-lg">{iconName.substring(0, 2).toUpperCase()}</span>;
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-2xl w-[480px] max-w-[90%] overflow-hidden">
        <div className="p-5 border-b border-slate-100 flex justify-between items-center">
          <div className="flex items-center">
            <div className={`w-10 h-10 rounded-lg ${tool.iconBg} flex items-center justify-center mr-3`}>
              {getIconComponent(tool.icon)}
            </div>
            <div>
              <h3 className="text-slate-800">{tool.name}</h3>
              <p className="text-xs text-slate-400">
                {tool.version || 'v1.0.0'} • {tool.author || 'By LangChain AI'}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* 名称（只读） */}
          <div>
            <label className="block text-xs text-slate-700 mb-1.5 uppercase tracking-wide">
              工具名称 (只读)
            </label>
            <input
              type="text"
              value={tool.name}
              readOnly
              className="w-full px-3 py-2.5 border border-slate-200 rounded-lg text-sm bg-slate-50 text-slate-500 cursor-not-allowed"
            />
          </div>

          {/* 描述（可编辑） */}
          <div>
            <label className="block text-xs text-slate-700 mb-1.5 uppercase tracking-wide">
              功能描述
            </label>
            <textarea
              value={description}
              onChange={(e) => {
                setDescription(e.target.value);
                setIsTested(false); // 修改后需要重新测试
              }}
              className="w-full h-20 px-3 py-2.5 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all resize-none"
              placeholder="描述这个工具的功能..."
            />
          </div>

          {/* 配置 */}
          <div>
            <label className="block text-xs text-slate-700 mb-1.5 uppercase tracking-wide flex items-center">
              工具配置 (JSON)
              <span className="ml-2 text-[10px] text-slate-400 normal-case">修改后需重新测试</span>
            </label>
            <textarea
              value={config}
              onChange={(e) => {
                setConfig(e.target.value);
                setIsTested(false); // 修改后需要重新测试
              }}
              className="w-full h-64 border border-slate-200 rounded-lg text-xs font-mono p-3 bg-slate-50 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all resize-none text-slate-600"
            />
          </div>
        </div>

        <div className="p-6 pt-4 border-t border-slate-100 flex justify-between items-center bg-slate-50/50">
          <div className="flex items-center space-x-3">
            {/* Test Connection Button */}
            <button
              onClick={handleTestConnection}
              disabled={isTesting}
              className={`text-xs flex items-center px-3 py-2 rounded transition-colors ${
                isTested 
                  ? 'text-green-600 bg-green-50' 
                  : 'text-green-600 hover:text-green-700 hover:bg-green-50'
              }`}
            >
              {isTesting ? (
                <>
                  <div className="w-3 h-3 border-2 border-green-600 border-t-transparent rounded-full animate-spin mr-1.5" />
                  测试中...
                </>
              ) : isTested ? (
                <>
                  <Check className="w-3 h-3 mr-1.5" />
                  已测试
                </>
              ) : (
                <>
                  <Plug className="w-3 h-3 mr-1.5" />
                  测试连接
                </>
              )}
            </button>
            {onDelete && (
              <button
                onClick={() => onDelete(tool.id)}
                className="text-xs text-red-600 hover:text-red-700 flex items-center px-2 py-1 hover:bg-red-50 rounded transition-colors"
              >
                <Trash2 className="w-3 h-3 mr-1.5" />
                删除工具
              </button>
            )}
          </div>
          <button
            onClick={handleSave}
            disabled={isSaving || !isTested}
            className="px-5 py-2.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors shadow-md shadow-indigo-200 flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSaved ? (
              <>
                <Check className="w-4 h-4 mr-1" />
                已保存
              </>
            ) : isSaving ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                保存中...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-1" />
                保存配置
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}