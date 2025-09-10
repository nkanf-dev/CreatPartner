# 🚀 CreatPartner

> AI驱动的创新创业项目助手 | AI-Powered Innovation & Entrepreneurship Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/database-MongoDB-green.svg)](https://mongodb.com/)

CreatPartner是一个专为创新创业项目设计的AI助手系统，特别针对参与创新创业竞赛的学生团队，提供智能的资料检索、项目规划、知识管理和决策支持。

## ✨ 核心特性

### 🤖 三Agent协作架构
- **主助手Agent** - 直接与用户交流，协调其他Agent
- **搜索Agent** - 自动进行外部资料检索（网络搜索 + ArXiv学术搜索）
- **知识Agent** - 管理项目长期记忆和知识库

### 🧠 双知识库系统
- **项目长期记忆库** - 存储项目发展历程、团队讨论、决策记录
- **外部资料库** - 整理从互联网和学术文献中检索的相关资料

### 🔍 智能检索能力
- **网络搜索** - 实时获取最新的行业动态和技术趋势
- **学术检索** - 从ArXiv等学术数据库获取前沿研究论文
- **语义搜索** - 基于Jina AI的向量检索和重排序技术

### 📋 项目管理功能
- **项目生命周期管理** - 从规划到部署的全阶段支持
- **任务智能分解** - 自动将复杂目标分解为可执行任务
- **进度跟踪** - 实时监控项目进展和里程碑

## 🛠️ 技术架构

### 核心技术栈
- **AI框架**: [Pydantic AI](https://ai.pydantic.org.cn/agents/) - 现代化的Python AI Agent框架
- **前端界面**: [Streamlit](https://streamlit.io/) - 快速构建AI应用界面
- **数据库**: [MongoDB](https://www.mongodb.com/) - 文档数据库存储项目数据
- **向量检索**: [Jina AI](https://docs.jina.ai/) - 嵌入向量生成和重排序
- **包管理**: [uv](https://docs.astral.sh/uv/) - 快速Python包管理器

### 系统架构图
```mermaid
graph TD
    A[用户界面 Streamlit] --> B[主助手 Agent]
    B --> C[搜索 Agent]
    B --> D[知识 Agent]
    
    C --> E[网络搜索 API]
    C --> F[ArXiv API]
    
    D --> G[项目记忆库]
    D --> H[外部资料库]
    
    G --> I[MongoDB + Jina Embedding]
    H --> I
    
    B --> J[项目管理]
    B --> K[任务规划]
    B --> L[决策支持]
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- MongoDB 4.4+
- 8GB+ RAM (推荐)

### 1. 安装项目
```bash
# 克隆项目
git clone https://github.com/nkanf-dev/CreatPartner.git
cd CreatPartner

# 安装依赖（使用uv，更快的包管理器）
pip install uv
uv sync
```

### 2. 配置环境
```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件
nano .env
```

**必需配置项**:
```bash
# LLM API密钥 (选择其中一个)
SILICONFLOW_API_KEY=your_api_key
# OPENAI_API_KEY=your_openai_key
# DEEPSEEK_API_KEY=your_deepseek_key

# MongoDB连接
MONGODB_URI=mongodb://localhost:27017
DB_NAME=creatpartner

# Jina AI API密钥
JINA_API_KEY=your_jina_api_key
```

### 3. 启动服务

#### 启动MongoDB
```bash
# 方式1: 直接启动 (如果已安装MongoDB)
mongod

# 方式2: 使用Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

#### 启动应用
```bash
# 启动Streamlit应用
uv run streamlit run app.py

# 或指定端口
uv run streamlit run app.py --server.port 8501
```

### 4. 访问应用
打开浏览器访问：`http://localhost:8501`

## 📖 使用指南

### 基本用法

1. **项目初始化**
   - 在侧边栏设置项目名称和描述
   - 选择项目当前阶段（规划/调研/开发/测试/部署/比赛）

2. **智能对话**
   - 在对话框中输入你的问题或需求
   - AI助手会自动判断需要执行的操作：
     - 🔍 搜索外部资料
     - 📚 查询知识库
     - 📋 制定计划
     - 📊 分析数据

3. **项目管理**
   - 查看和管理项目任务
   - 监控项目进度
   - 管理团队成员

4. **知识库管理**
   - 手动添加重要文档和资料
   - 搜索历史知识
   - 查看知识库统计

### 高级功能

#### 智能工作流程
CreatPartner支持预定义的工作流程，一键执行复杂任务：

- **研究工作流程**: 制定研究计划 → 搜索资料 → 整理知识库
- **分析工作流程**: 检索相关知识 → 综合分析 → 生成报告  
- **规划工作流程**: 制定详细计划 → 风险评估 → 资源分析

#### Agent协作示例
```
用户: "帮我调研AI在教育领域的应用现状"

主Agent: 分析请求 → 制定执行计划
├── 搜索Agent: 网络搜索最新教育AI产品和趋势
├── 搜索Agent: ArXiv搜索相关学术论文
└── 知识Agent: 整理资料到知识库

主Agent: 综合分析 → 生成调研报告
```

## ⚙️ 配置说明

### LLM提供商配置

#### 硅基流动 (推荐)
```bash
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your_key
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

#### OpenAI
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
LLM_MODEL=gpt-4-turbo-preview
```

#### DeepSeek
```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key
LLM_MODEL=deepseek-chat
```

### 数据库配置

#### MongoDB本地部署
```bash
MONGODB_URI=mongodb://localhost:27017
DB_NAME=creatpartner
```

#### MongoDB Atlas云服务
```bash
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=creatpartner
```

### 向量搜索配置
```bash
JINA_API_KEY=your_jina_key
JINA_EMBEDDING_MODEL=jina-embeddings-v3
JINA_RERANKER_MODEL=jina-reranker-v2-base-multilingual
VECTOR_SIMILARITY_THRESHOLD=0.7
```

## 🧪 开发指南

### 项目结构
```
CreatPartner/
├── app.py              # Streamlit主应用
├── config.py           # 配置管理
├── main_agent.py       # 主Agent逻辑
├── search_agent.py     # 搜索Agent
├── knowledge_agent.py  # 知识管理Agent
├── bp_reviewer_agent.py # 计划书审查Agent  
├── logger.py           # 日志系统
├── pyproject.toml      # 项目配置
├── uv.lock            # 依赖锁定文件
├── .env.example       # 环境配置模板
└── __pycache__/       # Python缓存文件
```

### 开发环境设置
```bash
# 安装开发依赖
uv sync --group dev

# 代码格式化
uv run black .
uv run isort .

# 类型检查
uv run mypy .

# 运行测试
uv run pytest
```

### 自定义Agent

1. **创建新Agent**
```python
from pydantic_ai import Agent
from config import config

my_agent = Agent(
    model=config.llm.model,
    system_prompt="你的自定义系统提示",
)

@my_agent.tool
async def my_custom_tool(query: str) -> str:
    """自定义工具功能"""
    # 实现你的逻辑
    return result
```

2. **集成到主系统**
```python
# 在main_agent.py中添加你的Agent
from your_agent import my_agent

class MainAgent:
    def __init__(self):
        self.my_agent = my_agent
```

### 扩展搜索能力

1. **添加新的搜索源**
```python
@search_agent.tool
async def custom_search(query: str) -> list:
    """自定义搜索源"""
    # 实现搜索逻辑
    return search_results
```

2. **自定义文档解析**
```python
# 使用MinerU解析文档
from mineru import DocumentParser

parser = DocumentParser()
content = parser.parse(file_path)
```

## 📊 监控与日志

### 性能监控
CreatPartner集成了Logfire监控系统：

```bash
# 启用监控
LOGFIRE_ENABLED=true
LOGFIRE_SEND=true

# 配置监控级别
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_METRICS=true
```

### 日志系统
- **实时日志**: 在Web界面查看系统运行状态
- **结构化日志**: JSON格式记录所有操作
- **性能指标**: 监控响应时间和资源使用

## 🚨 故障排除

### 常见问题

#### 1. MongoDB连接失败
```bash
# 检查MongoDB是否运行
mongosh --eval "db.runCommand({ping: 1})"

# 检查端口是否被占用
netstat -an | grep 27017
```

#### 2. API密钥无效
```bash
# 检查环境变量
echo $SILICONFLOW_API_KEY

# 测试API连接
curl -H "Authorization: Bearer $SILICONFLOW_API_KEY" \
     https://api.siliconflow.cn/v1/models
```

#### 3. 内存不足
```bash
# 检查系统资源
free -h
top

# 优化配置
LLM_MAX_TOKENS=4000
MAX_CONCURRENT_REQUESTS=2
```

#### 4. Streamlit启动失败
```bash
# 检查端口占用
lsof -i :8501

# 更换端口
uv run streamlit run app.py --server.port 8502
```

### 调试模式
启用调试模式获取详细错误信息：
```bash
DEBUG=true
LOG_LEVEL=DEBUG
STREAMLIT_SERVER_HEADLESS=false
```

## 🤝 贡献指南

### 参与贡献

1. **Fork项目**
2. **创建特性分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **提交更改**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. **推送到分支**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **提交Pull Request**

### 代码规范
- 使用Black进行代码格式化
- 遵循PEP 8编码标准
- 添加必要的类型注解
- 编写清晰的文档字符串

### 提交信息格式
```
type(scope): description

body (optional)

footer (optional)
```

类型说明：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具更新

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

## 🙏 致谢

- [Pydantic AI](https://ai.pydantic.org.cn/) - 强大的AI Agent框架
- [Streamlit](https://streamlit.io/) - 快速构建Web应用
- [MongoDB](https://www.mongodb.com/) - 灵活的文档数据库
- [Jina AI](https://jina.ai/) - 先进的向量搜索技术
- [MinerU](https://github.com/opendatalab/MinerU) - 优秀的文档解析工具

## 📞 联系我们

- **GitHub**: [nkanf-dev/CreatPartner](https://github.com/nkanf-dev/CreatPartner)
- **Issues**: [报告问题](https://github.com/nkanf-dev/CreatPartner/issues)
- **Discussions**: [参与讨论](https://github.com/nkanf-dev/CreatPartner/discussions)
- **Email**: team@creatpartner.ai

---

<div align="center">

**🚀 让AI助力你的创新创业之路！**

Made with ❤️ by CreatPartner Team

</div>
