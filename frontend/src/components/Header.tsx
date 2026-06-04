import { Button } from './ui/button';

interface HeaderProps {
  onChatClear: () => void;
  username: string;
  onLogout: () => void;
  isLoggingOut: boolean;
}

export function Header({ onChatClear, username, onLogout, isLoggingOut }: HeaderProps) {
  return (
    <header className="flex items-center justify-between p-4 bg-secondary border-b-2 border-black relative z-20">
      <div className="flex items-center">
        <Button 
          onClick={onChatClear}
          variant="outline"
          className="bg-white hover:bg-accent"
          disabled={isLoggingOut}
        >
          删除当前会话
        </Button>
      </div>
      
      <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-3">
        <h1 className="text-2xl font-bold text-black">
          Data Agent
        </h1>
        <span className="text-black text-sm font-medium">
          by 烨哲
        </span>
      </div>
      
      <div className="flex items-center gap-3">
        <div className="hidden rounded-full border border-black bg-white px-3 py-1 text-sm font-medium text-black md:block">
          {username}
        </div>
        <Button
          onClick={onLogout}
          variant="outline"
          className="bg-white hover:bg-accent"
          disabled={isLoggingOut}
        >
          {isLoggingOut ? '退出中...' : '退出登录'}
        </Button>
        <a 
          href="https://gitee.com/ye_sheng0839/data-agent" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center justify-center w-10 h-10"
        >
          <img 
            src="/gitee.svg" 
            alt="Gitee" 
            className="w-full h-full"
          />
        </a>
      </div>
    </header>
  );
}