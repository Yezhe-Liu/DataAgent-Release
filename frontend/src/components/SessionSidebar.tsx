import { MessageSquareText, Plus, Trash2 } from 'lucide-react';

import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { formatSessionTime, type StoredChatSession } from '../lib/chatSessions';

interface SessionSidebarProps {
  sessions: StoredChatSession[];
  activeSessionId: string;
  onCreateSession: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  disabled?: boolean;
  className?: string;
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
  disabled = false,
  className = '',
}: SessionSidebarProps) {
  return (
    <div className={`flex h-full min-h-0 flex-col bg-[#faf8f3] ${className}`}>
      <div className="border-b border-[#e7dfd1] p-4">
        <Button
          onClick={onCreateSession}
          variant="outline"
          className="h-11 w-full justify-center rounded-xl border-[#ded4c4] bg-white text-black shadow-sm hover:bg-[#fff7ed]"
          disabled={disabled}
        >
          <Plus className="w-4 h-4 mr-2" />
          新建会话
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-2 p-3">
          {sessions.length === 0 ? (
            <div className="rounded-xl border border-dashed border-[#d9cfbd] bg-white p-4 text-sm font-medium text-black/60 shadow-sm">
              暂无历史会话，发送第一条消息后会显示在这里。
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  role="button"
                  tabIndex={disabled ? -1 : 0}
                  onClick={() => {
                    if (!disabled) {
                      onSelectSession(session.id);
                    }
                  }}
                  onKeyDown={(event) => {
                    if (disabled) {
                      return;
                    }

                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      onSelectSession(session.id);
                    }
                  }}
                  className={`w-full rounded-xl border p-3 text-left transition-all ${
                    isActive
                      ? 'border-[#f0c677] bg-white shadow-sm ring-1 ring-[#f8dfab]'
                      : 'border-[#e6ded1] bg-[#fdfcf8] hover:bg-white hover:shadow-sm'
                  } ${disabled ? 'cursor-not-allowed opacity-70' : ''}`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border ${
                      isActive ? 'border-[#f0c677] bg-[#fff4dc]' : 'border-[#e6ded1] bg-white'
                    }`}>
                      <MessageSquareText className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div className="break-words text-sm font-semibold text-black">{session.title}</div>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onDeleteSession(session.id);
                          }}
                          disabled={disabled}
                          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[#e6ded1] bg-white hover:bg-[#fff1f1]"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="mt-2 min-h-[2.5rem] line-clamp-2 text-xs leading-5 text-black/60">
                        {session.preview || '暂无摘要'}
                      </div>
                      <div className="mt-2 text-[11px] font-medium text-black/45">
                        {formatSessionTime(session.updatedAt)}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
