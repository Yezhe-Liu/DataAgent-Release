import { useState } from 'react';

import { Button } from './ui/button';

interface AuthPanelProps {
  isLoading: boolean;
  error: string;
  onLogin: (username: string, password: string) => Promise<void>;
  onRegister: (username: string, password: string) => Promise<void>;
}

export function AuthPanel({ isLoading, error, onLogin, onRegister }: AuthPanelProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState('');

  const submit = async () => {
    const trimmedUsername = username.trim();
    if (trimmedUsername.length < 3) {
      setLocalError('用户名至少需要 3 个字符');
      return;
    }
    if (password.length < 6) {
      setLocalError('密码至少需要 6 个字符');
      return;
    }
    if (mode === 'register' && password !== confirmPassword) {
      setLocalError('两次输入的密码不一致');
      return;
    }

    setLocalError('');
    if (mode === 'login') {
      await onLogin(trimmedUsername, password);
      return;
    }
    await onRegister(trimmedUsername, password);
  };

  const currentError = localError || error;

  return (
    <div className="flex h-full items-center justify-center bg-[#f5f0e8] px-4">
      <div className="w-full max-w-md rounded-[32px] border border-[#e7dfd1] bg-white p-8 shadow-sm">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-black">Data Agent</h1>
          <div className="mt-2 text-sm text-black/55">登录后即可使用你的私有会话与数据分析空间</div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-2 rounded-2xl border border-[#e7dfd1] bg-[#fcfbf8] p-1">
          <button
            type="button"
            className={`rounded-2xl px-4 py-2 text-sm font-medium transition ${mode === 'login' ? 'bg-white shadow-sm' : 'text-black/55'}`}
            onClick={() => setMode('login')}
          >
            登录
          </button>
          <button
            type="button"
            className={`rounded-2xl px-4 py-2 text-sm font-medium transition ${mode === 'register' ? 'bg-white shadow-sm' : 'text-black/55'}`}
            onClick={() => setMode('register')}
          >
            注册
          </button>
        </div>

        <div className="mt-6 space-y-4">
          <div>
            <div className="mb-2 text-sm font-medium text-black">用户名</div>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="请输入用户名"
              className="w-full rounded-2xl border border-[#e7dfd1] px-4 py-3 text-sm outline-none transition focus:border-[#f0c677]"
              disabled={isLoading}
            />
          </div>

          <div>
            <div className="mb-2 text-sm font-medium text-black">密码</div>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="请输入密码"
              className="w-full rounded-2xl border border-[#e7dfd1] px-4 py-3 text-sm outline-none transition focus:border-[#f0c677]"
              disabled={isLoading}
            />
          </div>

          {mode === 'register' && (
            <div>
              <div className="mb-2 text-sm font-medium text-black">确认密码</div>
              <input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder="请再次输入密码"
                className="w-full rounded-2xl border border-[#e7dfd1] px-4 py-3 text-sm outline-none transition focus:border-[#f0c677]"
                disabled={isLoading}
              />
            </div>
          )}
        </div>

        {currentError && (
          <div className="mt-4 rounded-2xl border border-[#f1c7c7] bg-[#fff4f4] px-4 py-3 text-sm text-[#b42318]">
            {currentError}
          </div>
        )}

        <Button
          onClick={() => void submit()}
          disabled={isLoading}
          className="mt-6 h-12 w-full rounded-2xl border-[#f0c677] bg-[#ffd166] text-black hover:bg-[#ffc94a]"
        >
          {isLoading ? '处理中...' : mode === 'login' ? '登录' : '注册并登录'}
        </Button>
      </div>
    </div>
  );
}
