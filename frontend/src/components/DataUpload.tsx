import { useRef, useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import { Button } from './ui/button';
import { Plus, Trash2, Upload } from 'lucide-react';
import { TelcoData } from '../data/mockData';
import { API_ENDPOINTS, apiFetch } from '../config/api';

interface DataUploadProps {
  onDataUpload: (data: TelcoData[]) => void;
  onDataClear: () => void;
  uploadedData: TelcoData[] | null;
  disabled?: boolean;
}

export function DataUpload({ onDataUpload, onDataClear, uploadedData, disabled = false }: DataUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const openFilePicker = () => {
    if (disabled) {
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (file: File) => {
    if (disabled) {
      return;
    }
    if (!file.name.endsWith('.csv')) {
      alert('请上传CSV格式的文件');
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiFetch(API_ENDPOINTS.UPLOAD, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === 'success' && result.preview && Array.isArray(result.preview)) {
        onDataUpload(result.preview);
        if (result.message) {
          alert(result.message);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('文件上传失败，请检查后端服务是否正常运行');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void handleFileSelect(file);
    }

    e.target.value = '';
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) {
      return;
    }

    const file = e.dataTransfer.files?.[0];
    if (file) {
      void handleFileSelect(file);
    }
  };

  const handleClearData = async () => {
    onDataClear();
  };

  const titleText = uploadedData ? '更新数据集' : '上传数据集';
  const descriptionText = uploadedData
    ? `当前已加载 ${uploadedData.length} 行数据，点击加号或拖拽 CSV 可快速替换。`
    : disabled
      ? '登录后可上传并管理你的私有 CSV 数据集。'
      : '点击加号选择 CSV 文件，或直接拖拽到这里上传。';
  const statusText = isUploading
    ? '上传中...'
    : disabled
      ? '登录后可用'
      : uploadedData
        ? '已连接数据集'
        : '尚未上传数据';

  return (
    <div className="bg-[#fcfbf8] px-3 py-3 sm:px-4">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileInputChange}
        className="hidden"
      />

      <div
        className={`flex min-h-[88px] items-center gap-3 rounded-[24px] border px-3 py-3 transition-all duration-150 sm:px-4 ${
          isDragOver
            ? 'border-[#f0c677] bg-[#fff4dc] shadow-sm'
            : 'border-[#e7dfd1] bg-white shadow-sm'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <Button
          onClick={openFilePicker}
          disabled={isUploading || disabled}
          size="icon"
          className="h-12 w-12 rounded-2xl border-[#f0c677] bg-[#ffd166] text-black hover:bg-[#ffc94a]"
        >
          <Plus className="h-5 w-5" />
        </Button>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <div className="truncate text-sm font-semibold text-black">{titleText}</div>
            <div className="inline-flex items-center gap-1 rounded-full border border-[#efe6d7] bg-[#fcfbf8] px-2 py-1 text-[11px] font-medium text-[#946200]">
              <Upload className="h-3.5 w-3.5" />
              {statusText}
            </div>
          </div>

          <div className="mt-1 text-xs leading-5 text-black/55">
            {isDragOver ? '松开鼠标即可开始上传或替换当前 CSV。' : descriptionText}
          </div>
        </div>

        {uploadedData && (
          <Button
            onClick={handleClearData}
            variant="outline"
            size="icon"
            className="h-10 w-10 rounded-xl border-[#ecd2d2] bg-white hover:bg-[#fff1f1]"
            disabled={disabled}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}