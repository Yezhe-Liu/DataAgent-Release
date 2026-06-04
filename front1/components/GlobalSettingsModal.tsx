import { X, Settings, Cpu, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { getLLMConfig, updateLLMConfig, LLMProfile } from '../utils/llmConfigApi';

interface GlobalSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const ROLE_LABELS: { key: string; label: string; description: string }[] = [
  { key: 'router', label: '路由 (意图分析)', description: '判断用户问题属于直答 / 知识库 / 工具任务。' },
  { key: 'planner', label: '规划 (工具编排)', description: '根据意图生成工具调用步骤。' },
  { key: 'answer', label: '回答合成', description: '基于上下文与工具结果生成最终回复。' },
];

export function GlobalSettingsModal({ isOpen, onClose }: GlobalSettingsModalProps) {
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [debugMode, setDebugMode] = useState(false);
  const [allowLocalFiles, setAllowLocalFiles] = useState(true);

  const [profiles, setProfiles] = useState<LLMProfile[]>([]);
  const [defaultProfile, setDefaultProfile] = useState<string>('');
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [loadingConfig, setLoadingConfig] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [configError, setConfigError] = useState<string>('');

  useEffect(() => {
    if (!isOpen) return;
    let aborted = false;
    const loadConfig = async () => {
      setLoadingConfig(true);
      setConfigError('');
      try {
        const payload = await getLLMConfig();
        if (aborted) return;
        setProfiles(payload.profiles || []);
        setDefaultProfile(payload.default_profile || '');
        setOverrides(payload.profile_overrides || {});
      } catch (err) {
        if (aborted) return;
        setConfigError(err instanceof Error ? err.message : '未能加载 LLM 配置');
      } finally {
        if (!aborted) setLoadingConfig(false);
      }
    };
    loadConfig();
    return () => {
      aborted = true;
    };
  }, [isOpen]);

  const handleOverrideChange = (role: string, value: string) => {
    setOverrides((prev) => {
      const next = { ...prev };
      if (!value) {
        delete next[role];
      } else {
        next[role] = value;
      }
      return next;
    });
  };

  const handleSaveLLM = async () => {
    if (!defaultProfile) {
      toast.error('请先选择默认模型');
      return;
    }
    setSavingConfig(true);
    try {
      const payload = await updateLLMConfig({
        default_profile: defaultProfile,
        profile_overrides: overrides,
      });
      setProfiles(payload.profiles || []);
      setDefaultProfile(payload.default_profile || '');
      setOverrides(payload.profile_overrides || {});
      toast.success('模型配置已保存', {
        description: `默认模型：${payload.default_profile}`,
        duration: 3000,
      });
    } catch (err) {
      toast.error('保存失败', {
        description: err instanceof Error ? err.message : '无法更新模型配置',
        duration: 4000,
      });
    } finally {
      setSavingConfig(false);
    }
  };

  const renderProfileOption = (profile: LLMProfile) => {
    const unavailable = !profile.api_key_available;
    const label = `${profile.name} · ${profile.provider} / ${profile.model || '—'}${unavailable ? ' (缺少 API Key)' : ''}`;
    return (
      <option key={profile.name} value={profile.name} disabled={unavailable}>
        {label}
      </option>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-2xl w-[560px] max-w-[92%] overflow-hidden flex flex-col max-h-[90vh]">
        <div className="p-5 border-b border-slate-100 flex justify-between items-center flex-shrink-0">
          <h3 className="text-slate-800 text-lg flex items-center">
            <Settings className="w-5 h-5 text-slate-400 mr-2" />
            全局设置
          </h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-5 overflow-y-auto">
          {/* LLM 模型配置 */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-4 h-4 text-indigo-500" />
              <span className="text-sm font-medium text-slate-800">LLM 模型配置</span>
            </div>

            {loadingConfig ? (
              <div className="flex items-center gap-2 text-sm text-slate-500 py-4">
                <Loader2 className="w-4 h-4 animate-spin" />
                正在加载模型列表...
              </div>
            ) : configError ? (
              <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>{configError}</span>
              </div>
            ) : (
              <div className="space-y-3">
                {/* 默认模型 */}
                <div>
                  <label className="block text-xs text-slate-600 mb-1">默认模型（未指定角色时使用）</label>
                  <select
                    value={defaultProfile}
                    onChange={(e) => setDefaultProfile(e.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 bg-white focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200"
                  >
                    {profiles.map(renderProfileOption)}
                  </select>
                </div>

                {/* 角色覆盖 */}
                <div className="pt-2 border-t border-slate-100">
                  <div className="text-xs text-slate-500 mb-2">按角色覆盖（可选）</div>
                  <div className="space-y-2">
                    {ROLE_LABELS.map(({ key, label, description }) => (
                      <div key={key} className="flex items-center gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="text-xs text-slate-700">{label}</div>
                          <div className="text-[11px] text-slate-400 truncate">{description}</div>
                        </div>
                        <select
                          value={overrides[key] || ''}
                          onChange={(e) => handleOverrideChange(key, e.target.value)}
                          className="flex-shrink-0 w-[180px] px-2 py-1.5 text-xs rounded-md border border-slate-200 bg-white focus:outline-none focus:border-indigo-400"
                        >
                          <option value="">使用默认</option>
                          {profiles.map(renderProfileOption)}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Profile 状态提示 */}
                {profiles.some((p) => !p.api_key_available) && (
                  <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-700">
                    <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                    <span>部分模型缺少 API Key（请在后端 .env 中配置对应的环境变量），已禁用选择。</span>
                  </div>
                )}

                <button
                  onClick={handleSaveLLM}
                  disabled={savingConfig}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed transition-colors"
                >
                  {savingConfig ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
                  {savingConfig ? '保存中...' : '保存模型配置'}
                </button>
              </div>
            )}
          </section>

          {/* MCP 全局开关 */}
          <section className="pt-3 border-t border-slate-100 space-y-2">
            <div className="text-sm font-medium text-slate-800 mb-1">MCP 设置</div>

            <div className="flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 transition-colors">
              <div>
                <div className="text-sm text-slate-800">自动更新工具</div>
                <div className="text-xs text-slate-400">保持所有 MCP 工具为最新版本</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoUpdate}
                  onChange={(e) => setAutoUpdate(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 transition-colors">
              <div>
                <div className="text-sm text-slate-800">Debug 模式</div>
                <div className="text-xs text-slate-400">在控制台输出详细的 MCP 通信日志</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={debugMode}
                  onChange={(e) => setDebugMode(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 transition-colors">
              <div>
                <div className="text-sm text-slate-800">允许本地文件访问</div>
                <div className="text-xs text-slate-400">授权 FileSystem MCP 读取本地文件</div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={allowLocalFiles}
                  onChange={(e) => setAllowLocalFiles(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
              </label>
            </div>
          </section>
        </div>

        <div className="p-4 border-t border-slate-100 flex justify-end flex-shrink-0">
          <button
            onClick={onClose}
            className="px-5 py-2 bg-slate-900 text-white text-sm rounded-lg hover:bg-slate-800 transition-colors"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}
