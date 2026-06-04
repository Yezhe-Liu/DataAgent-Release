import { useEffect, useState } from 'react';
import { AuthPanel } from './components/AuthPanel';
import { Header } from './components/Header';
import { ChatWorkspace } from './components/ChatWorkspace';
import { DataUpload } from './components/DataUpload';
import { VisualizationPanel } from './components/VisualizationPanel';
import { TelcoData } from './data/mockData';
import { fetchCurrentUser, login, logout, register, type AuthUser } from './lib/auth';

export default function App() {
  const [uploadedData, setUploadedData] = useState<TelcoData[] | null>(null);
  const [chatClearTrigger, setChatClearTrigger] = useState(0);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authError, setAuthError] = useState('');
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  useEffect(() => {
    let isCancelled = false;

    const loadCurrentUser = async () => {
      setAuthLoading(true);
      try {
        const user = await fetchCurrentUser();
        if (isCancelled) {
          return;
        }
        setCurrentUser(user);
        setAuthError('');
      } catch (error) {
        if (isCancelled) {
          return;
        }
        setCurrentUser(null);
        setAuthError(error instanceof Error ? error.message : '认证状态加载失败');
      } finally {
        if (!isCancelled) {
          setAuthLoading(false);
        }
      }
    };

    void loadCurrentUser();

    return () => {
      isCancelled = true;
    };
  }, []);

  useEffect(() => {
    setUploadedData(null);
    setGeneratedImage(null);
  }, [currentUser?.id]);

  const handleDataUpload = (data: TelcoData[]) => {
    setUploadedData(data);
  };

  const handleDataClear = () => {
    setUploadedData(null);
  };

  const handleChatClear = () => {
    setChatClearTrigger(prev => prev + 1);
    setGeneratedImage(null);
  };

  const handleImageGenerated = (imageUrl: string | null) => {
    setGeneratedImage(imageUrl);
  };

  const handleLogin = async (username: string, password: string) => {
    setAuthLoading(true);
    setAuthError('');
    try {
      const user = await login(username, password);
      setCurrentUser(user);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : '登录失败');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleRegister = async (username: string, password: string) => {
    setAuthLoading(true);
    setAuthError('');
    try {
      const user = await register(username, password);
      setCurrentUser(user);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : '注册失败');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await logout();
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : '退出登录失败');
    } finally {
      setCurrentUser(null);
      setUploadedData(null);
      setGeneratedImage(null);
      setChatClearTrigger((prev) => prev + 1);
      setIsLoggingOut(false);
    }
  };

  if (authLoading && !currentUser) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#f5f0e8] text-black/65">
        正在检查登录状态...
      </div>
    );
  }

  if (!currentUser) {
    return (
      <AuthPanel
        isLoading={authLoading}
        error={authError}
        onLogin={handleLogin}
        onRegister={handleRegister}
      />
    );
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#f5f0e8]">
      <div className="flex flex-col h-full">
        <Header
          onChatClear={handleChatClear}
          username={currentUser.username}
          onLogout={() => void handleLogout()}
          isLoggingOut={isLoggingOut}
        />
        
        <div className="flex-1 grid overflow-hidden min-h-0 grid-cols-1 lg:grid-cols-[minmax(560px,1.18fr)_minmax(360px,0.82fr)]">
          {/* Left Panel - Chat Interface (Area 1) */}
          <div className="min-w-0 flex flex-col min-h-0 bg-white overflow-hidden lg:border-b-0 lg:border-r-2">
            <ChatWorkspace
              clearTrigger={chatClearTrigger}
              userId={currentUser.id}
              onImageGenerated={handleImageGenerated}
            />
          </div>
          
          {/* Right Panel - Data Analysis (Area 2 & 3) */}
          <div className="min-w-0 flex flex-col min-h-0">
            {/* Top Half - Visualization (Area 2) */}
            <div className="min-h-0 flex-1 overflow-hidden" style={{ backgroundColor: '#F0F0F0' }}>
              <VisualizationPanel uploadedData={uploadedData} generatedImage={generatedImage} />
            </div>
            
            {/* Bottom Half - Data Upload/Display (Area 3) */}
            <div className="border-t-2 border-black" style={{ backgroundColor: '#F7F7F5' }}>
              <DataUpload
                onDataUpload={handleDataUpload}
                onDataClear={handleDataClear}
                uploadedData={uploadedData}
                disabled={!currentUser}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}