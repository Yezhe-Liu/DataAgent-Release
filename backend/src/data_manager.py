import pandas as pd
import numpy as np
import io
import os  # 需要导入 os 来获取文件名
from dataclasses import dataclass
from threading import Lock
from src.runtime_context import get_current_user_id

# 全局变量存储
@dataclass
class DatasetState:
    dataframe: pd.DataFrame | None = None
    filename: str = "未命名数据集"

DATASET_STATES: dict[str, DatasetState] = {}
DATASET_LOCK = Lock()

def _resolve_user_key(user_id: str | None = None) -> str:
    candidate = (user_id or get_current_user_id() or "").strip()
    return candidate or "__global__"

def _get_dataset_state(user_id: str | None = None) -> DatasetState:
    resolved_user_id = _resolve_user_key(user_id)
    with DATASET_LOCK:
        state = DATASET_STATES.get(resolved_user_id)
        if state is None:
            state = DatasetState()
            DATASET_STATES[resolved_user_id] = state
        return state

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    [内部函数] 数据预处理流水线：
    1. 去除全空行列
    2. 智能类型推断 (object -> numeric)
    3. 缺失值填充
    """
    # 1. 基础清理：去除全空的行和列
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # 2. 智能类型转换
    # 尝试将 object 类型的列转换为数值，无法转换的变成 NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # 尝试转数字，coerce 模式下无法转换的变为 NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # 如果转换后的非空值比例超过 50%，我们认为这列应该是数字列 (例如: "10", "20", "N/A")
                if numeric_series.notna().sum() > 0.5 * len(df):
                    df[col] = numeric_series
            except Exception:
                pass

    # 3. 缺失值处理 (简单粗暴策略，适合演示)
    # 数值列：用均值填充
    # 类别列：用 "Unknown" 填充
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df

def load_csv_file(file_path: str, user_id: str | None = None):
    """加载并清洗 CSV 文件"""
    try:
        state = _get_dataset_state(user_id)
        state.filename = os.path.basename(file_path)
        
        raw_df = pd.read_csv(file_path)
        clean_df = _preprocess_data(raw_df)
        state.dataframe = clean_df
        
        rows, cols = clean_df.shape
        
        return True, (f"成功加载文件【{state.filename}】！\n"
                      f"包含 {rows} 行，{cols} 列。")
    except Exception as e:
        return False, f"数据加载失败: {str(e)}"

def get_dataframe(user_id: str | None = None):
    """获取当前 DataFrame"""
    return _get_dataset_state(user_id).dataframe

def get_data_preview(n=10, user_id: str | None = None):
    """获取前 N 行数据 (处理 NaN 为 None 以便 JSON 序列化)"""
    dataframe = get_dataframe(user_id)
    if dataframe is not None:
        # replace({np.nan: None}) 是为了防止前端 JSON 解析报错
        return dataframe.head(n).replace({np.nan: None}).to_dict(orient='records')
    return []

def get_data_info(user_id: str | None = None):
    """获取数据摘要 (Schema)"""
    state = _get_dataset_state(user_id)
    if state.dataframe is not None:
        buffer = io.StringIO()
        state.dataframe.info(buf=buffer)
        
        # [修改] 在返回的信息头部拼接文件名
        info_str = f"数据来源文件: {state.filename}\n" 
        info_str += "-" * 30 + "\n"
        info_str += buffer.getvalue()
        
        return info_str
    return "暂无数据"

def calculate_correlation(col1: str, col2: str, user_id: str | None = None):
    """
    [增强版] 计算相关性
    支持：数值 vs 数值 (Pearson), 类别 vs 数值 (Label Encoding), 类别 vs 类别
    """
    dataframe = get_dataframe(user_id)
    if dataframe is None:
        return 0.0

    if col1 not in dataframe.columns or col2 not in dataframe.columns:
        return 0.0

    try:
        s1 = dataframe[col1]
        s2 = dataframe[col2]

        # 辅助函数：将序列转为数值
        def to_numeric_force(series):
            if pd.api.types.is_numeric_dtype(series):
                return series
            else:
                # 如果是字符串/类别，使用 factorize 编码 (0, 1, 2...)
                codes, uniques = pd.factorize(series)
                return pd.Series(codes)

        # 强制转换为数值序列
        v1 = to_numeric_force(s1)
        v2 = to_numeric_force(s2)

        # 计算 Pearson 相关系数
        corr = v1.corr(v2)
        
        # 处理计算结果为 NaN 的情况 (例如标准差为0)
        if pd.isna(corr):
            return 0.0
            
        return round(corr, 4)

    except Exception as e:
        print(f"相关性计算出错: {e}")
        return 0.0