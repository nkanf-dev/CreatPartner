"""
CreatPartner 配置文件
集中管理所有配置项，支持自定义LLM服务提供商
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class LLMConfig:
    """LLM配置"""
    # 基础配置
    provider: str = "siliconflow"  # 默认使用硅基流动
    model_name: str = "deepseek-ai/DeepSeek-V2.5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # 请求配置
    max_retries: int = 3
    timeout: float = 60.0
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 根据提供商设置默认值
        if self.provider == "siliconflow":
            self.api_key = self.api_key or os.getenv("SILICONFLOW_API_KEY")
            self.base_url = self.base_url or "https://api.siliconflow.cn/v1"
        elif self.provider == "openai":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.base_url = self.base_url or "https://api.openai.com/v1"
        elif self.provider == "deepseek":
            self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = self.base_url or "https://api.deepseek.com/v1"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "creatpartner"
    
    def __post_init__(self):
        """初始化后处理"""
        self.mongodb_uri = os.getenv("MONGODB_URI", self.mongodb_uri)
        self.database_name = os.getenv("DB_NAME", self.database_name)


@dataclass
class EmbeddingConfig:
    """嵌入服务配置"""
    provider: str = "jina"
    api_key: Optional[str] = None
    model: str = "jina-embeddings-v3"
    dimensions: int = 1024
    
    def __post_init__(self):
        """初始化后处理"""
        if self.provider == "jina":
            self.api_key = self.api_key or os.getenv("JINA_API_KEY")


@dataclass
class SearchConfig:
    """搜索配置"""
    max_results: int = 5
    enable_deep_search: bool = False
    enable_content_extraction: bool = True
    enable_reranker: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        self.max_results = int(os.getenv("MAX_SEARCH_RESULTS", self.max_results))
        self.enable_deep_search = os.getenv("ENABLE_DEEP_SEARCH", "false").lower() == "true"


@dataclass
class ProjectConfig:
    """项目配置"""
    default_project_name: str = "未命名项目"
    default_project_stage: Literal["planning", "research", "development", "testing", "deployment"] = "planning"
    default_project_description: str = ""
    
    # 使用限制
    max_requests_per_session: int = 20
    max_tokens_per_session: int = 10000
    max_tool_calls_per_session: int = 10


@dataclass
class AppConfig:
    """应用配置"""
    # 基础配置
    app_name: str = "CreatPartner"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # 组件配置
    llm: LLMConfig = None
    database: DatabaseConfig = None
    embedding: EmbeddingConfig = None
    search: SearchConfig = None
    project: ProjectConfig = None
    
    def __post_init__(self):
        """初始化后处理"""
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # 初始化子配置
        if self.llm is None:
            self.llm = LLMConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.search is None:
            self.search = SearchConfig()
        if self.project is None:
            self.project = ProjectConfig()


# 全局配置实例
config = AppConfig()


def get_model_name() -> str:
    """获取完整的模型名称（包含提供商前缀）"""
    if config.llm.provider == "siliconflow":
        return f"openai:{config.llm.model_name}"
    elif config.llm.provider == "openai":
        return f"openai:{config.llm.model_name}"
    elif config.llm.provider == "deepseek":
        return f"deepseek:{config.llm.model_name}"
    else:
        return f"openai:{config.llm.model_name}"


def create_llm_provider():
    """创建LLM提供商实例"""
    try:
        from pydantic_ai.providers.openai import OpenAIProvider
        from openai import AsyncOpenAI
        
        # 创建自定义客户端
        client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            max_retries=config.llm.max_retries,
            timeout=config.llm.timeout
        )
        
        return OpenAIProvider(openai_client=client)
        
    except ImportError:
        print("警告: 缺少依赖包，返回默认提供商")
        return None


def validate_config() -> bool:
    """验证配置的有效性"""
    errors = []
    
    # 检查必需的API密钥
    if not config.llm.api_key:
        errors.append(f"缺少{config.llm.provider.upper()}_API_KEY环境变量")
    
    if config.embedding.provider == "jina" and not config.embedding.api_key:
        errors.append("缺少JINA_API_KEY环境变量")
    
    # 检查数据库连接
    if not config.database.mongodb_uri:
        errors.append("缺少MongoDB连接配置")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def print_config():
    """打印当前配置（隐藏敏感信息）"""
    print(f"CreatPartner 配置:")
    print(f"  LLM: {config.llm.provider} ({config.llm.model_name})")
    print(f"  API密钥: {'已配置' if config.llm.api_key else '未配置'}")
    print(f"  Base URL: {config.llm.base_url}")
    print(f"  数据库: {config.database.mongodb_uri}")
    print(f"  嵌入服务: {config.embedding.provider}")
    print(f"  调试模式: {config.debug}")


if __name__ == "__main__":
    print_config()
    print(f"\n配置有效性: {'✅ 通过' if validate_config() else '❌ 失败'}")