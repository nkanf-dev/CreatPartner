# CreatPartner 多代理架构部署指南

## 🏗️ 架构概述

CreatPartner 基于 Pydantic AI 多代理设计模式重新架构，实现了更加模块化、可扩展和高效的 AI 研究助手系统。

### 架构特点

1. **智能体委托 (Agent Delegation)**
   - 主代理将专门任务委托给特定的子代理
   - 自动选择最佳的处理策略
   - 统一的使用量统计和限制管理

2. **程序化智能体交接 (Programmatic Agent Hand-off)**
   - 结构化的工作流程
   - 明确的代理责任分工
   - 优雅的错误处理和恢复

3. **共享依赖管理**
   - 统一的配置和资源管理
   - 高效的连接复用
   - 灵活的环境配置

## 🚀 快速部署

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd CreatPartner

# 安装依赖
python -m pip install uv
uv pip install -r requirements.txt

# 或使用安装脚本
python install.py
```

### 2. 环境变量配置

创建 `.env` 文件：

```env
# 必需的API密钥
OPENAI_API_KEY=your_openai_api_key_here
JINA_API_KEY=your_jina_api_key_here

# 数据库配置
MONGODB_URI=mongodb://localhost:27017
# 或 MongoDB Atlas
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# 数据库名称
DB_NAME=creatpartner

# 可选配置
DEFAULT_MODEL=openai:gpt-4o
MAX_SEARCH_RESULTS=5
MAX_KNOWLEDGE_RESULTS=10
```

### 3. 代理组件说明

#### ResearchCoordinator (研究协调器)
```python
# 负责智能体委托
coordinator = ResearchCoordinator()

# 委托搜索任务
search_result = await coordinator.delegate_search_task(
    ctx, "AI技术趋势", "web", True
)

# 委托知识查询
knowledge_result = await coordinator.delegate_knowledge_query(
    ctx, "项目风险分析", "project_memory"
)
```

#### CreatPartnerAgent (主代理)
```python
# 程序化智能体交接
agent = create_creatpartner_agent()

# 研究工作流
result = await agent.research_workflow(
    "市场竞争分析", shared_deps
)

# 持续对话
response = await agent.chat("下一步应该做什么？", shared_deps)
```

### 4. 部署方式

#### 方式1: Web应用 (推荐)
```bash
# 启动Streamlit应用
streamlit run app.py

# 访问地址
# http://localhost:8501
```

#### 方式2: 命令行使用
```python
from main_agent import create_creatpartner_agent, create_shared_dependencies

# 创建代理
agent = create_creatpartner_agent()

# 配置依赖
deps = create_shared_dependencies(
    project_name="我的创新项目",
    project_description="项目描述",
    project_stage="research"
)

# 开始会话
welcome = await agent.start_project_session(
    "项目名称", "项目描述", "research"
)

# 进行对话
response = await agent.chat("帮我分析市场趋势", deps)
```

#### 方式3: API服务
```python
# 可以基于FastAPI创建REST API
from fastapi import FastAPI
from main_agent import create_creatpartner_agent

app = FastAPI()
agent = create_creatpartner_agent()

@app.post("/chat")
async def chat_endpoint(message: str, project_config: dict):
    deps = create_shared_dependencies(**project_config)
    response = await agent.chat(message, deps)
    return {"response": response}
```

## 🧪 测试验证

### 单元测试
```bash
# 测试多代理架构
python test_multi_agent.py

# 测试Jina AI集成
python test_jina_integration.py
```

### 功能验证
```python
# 验证智能体委托
await agent.research_coordinator.delegate_search_task(
    ctx, "测试查询", "both", True
)

# 验证程序化交接
result = await agent.research_workflow("测试工作流", deps)

# 验证知识库集成
knowledge_stats = await agent.research_coordinator.delegate_knowledge_query(
    ctx, "统计信息", None
)
```

## 🔧 配置优化

### 使用限制配置
```python
deps = create_shared_dependencies(
    project_name="项目名称",
    request_limit=20,          # 最大请求数
    total_tokens_limit=10000,  # 最大token数
    tool_calls_limit=10        # 最大工具调用数
)
```

### 性能优化
```python
# 调整搜索结果数量
deps.max_search_results = 3

# 调整知识库查询结果
deps.max_knowledge_results = 5

# 启用重排序优化
use_reranker = True
```

### 错误处理
```python
try:
    result = await agent.research_workflow(query, deps)
except Exception as e:
    # 优雅降级
    fallback_result = await agent.chat(
        f"请基于现有知识回答: {query}", deps
    )
```

## 📊 监控和维护

### 使用量监控
```python
# 检查使用统计
print(f"请求数: {result.usage().requests}")
print(f"Token数: {result.usage().total_tokens}")
print(f"工具调用: {result.usage().tool_calls}")
```

### 知识库维护
```bash
# 创建向量索引（首次部署）
python -c "
from knowledge_agent import KnowledgeAgent, KnowledgeDependencies
import os

agent = KnowledgeAgent()
deps = KnowledgeDependencies(
    mongodb_uri=os.getenv('MONGODB_URI'),
    database_name=os.getenv('DB_NAME', 'creatpartner')
)
agent.create_vector_search_index(deps)
print('✅ 向量索引创建完成')
"
```

### 日志和调试
```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 在代理中添加日志
logger.info(f"开始处理查询: {query}")
logger.info(f"使用代理: {agent.__class__.__name__}")
```

## 🔄 升级和扩展

### 添加新的代理
```python
class CustomAgent:
    def __init__(self, model_name: str):
        self.agent = Agent(
            model_name,
            deps_type=SharedDependencies,
            system_prompt="自定义代理系统提示"
        )
    
    @self.agent.tool
    async def custom_task(self, ctx: RunContext[SharedDependencies]):
        # 自定义任务逻辑
        pass

# 在协调器中注册
coordinator.custom_agent = CustomAgent()
```

### 扩展工作流
```python
async def extended_workflow(self, query: str, deps: SharedDependencies):
    # 1. 原有工作流
    base_result = await self.research_workflow(query, deps)
    
    # 2. 扩展处理
    enhanced_result = await self.custom_processing(base_result, deps)
    
    return enhanced_result
```

## 📋 故障排除

### 常见问题

1. **API密钥错误**
   ```
   确保.env文件中正确配置了所有必需的API密钥
   ```

2. **MongoDB连接失败**
   ```
   检查MONGODB_URI是否正确
   确保MongoDB服务正在运行
   ```

3. **向量索引问题**
   ```
   重新创建向量索引：
   python -c "from knowledge_agent import KnowledgeAgent; agent = KnowledgeAgent(); agent.create_vector_search_index(deps)"
   ```

4. **使用量超限**
   ```
   调整UsageLimits配置：
   deps.usage_limits = UsageLimits(request_limit=50)
   ```

### 调试工具
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试单个组件
python test_multi_agent.py

# 验证Jina服务
python test_jina_integration.py
```

## 🎯 最佳实践

1. **合理设置使用限制**，避免API成本失控
2. **定期备份知识库**，保护项目数据
3. **监控代理性能**，及时优化配置
4. **保持代理系统提示更新**，提升响应质量
5. **使用结构化的项目配置**，便于管理多个项目

---

通过以上配置，您就可以成功部署和使用基于多代理架构的CreatPartner系统了！
