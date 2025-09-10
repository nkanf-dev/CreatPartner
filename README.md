# CreatPartner - AI驱动的创新竞赛助手

基于 Pydantic AI 多代理架构开发的智能研究助手，专为大学生创新创业竞赛提供全方位的信息检索、知识管理和项目分析支持。

## 🎯 核心功能

### 多代理架构设计

基于Pydantic AI最佳实践，采用**智能体委托**和**程序化智能体交接**模式：

#### 1. 研究协调器 (ResearchCoordinator)
- **智能体委托**: 自动选择合适的子代理处理任务
- **搜索代理委托**: 委托网络和学术搜索任务
- **知识代理委托**: 委托知识存储和查询任务
- **工作流协调**: 确保信息在不同代理间正确传递

#### 2. 用户交互代理 (InteractionAgent)
- **主要接口**: 用户的直接交互界面
- **需求理解**: 分析和澄清用户需求
- **策略制定**: 自动选择最佳的研究和分析策略
- **结果整合**: 汇总多个代理的输出并提供统一回答

#### 3. 搜索代理 (SearchAgent)
- **Web搜索**: 基于Jina API的网络信息检索
- **学术搜索**: 基于Arxiv的论文检索
- **智能分析**: AI驱动的搜索结果分析和总结

#### 4. 知识库代理 (KnowledgeAgent)
- **项目长期记忆**: 存储项目历史、决策、经验教训
- **外部检索资料**: 整理和归档从外部获取的研究资料
- **向量搜索**: 基于MongoDB Atlas Vector Search和Jina AI的语义检索
- **智能分割**: 使用Jina Segmenter自动分割长文档
- **重排序优化**: 使用Jina Reranker提升搜索相关性

### 程序化智能体交接工作流

```
用户查询 → 知识库检查 → 外部搜索(如需) → 信息整合 → 智能回答
     ↓            ↓            ↓           ↓         ↓
用户交互代理 → 知识代理 → 搜索代理 → 研究协调器 → 用户交互代理
```

### 两个知识库

#### 项目长期记忆知识库
- 项目决策记录
- 开发进展跟踪
- 经验教训总结
- 团队协作记录

#### 外部检索资料知识库
- Web搜索结果
- 学术论文资料
- 行业报告分析
- 竞品调研信息

## 🚀 技术栈

- **Agent框架**: Pydantic AI (多代理架构)
- **向量数据库**: MongoDB Atlas Vector Search
- **AI服务**: Jina AI (嵌入、重排序、分割)
  - Jina Embeddings v3 (1024维向量)
  - Jina Reranker v2 (多语言重排序)
  - Jina Segmenter (智能文本分割)
- **网络搜索**: Jina Search API
- **学术搜索**: Arxiv API
- **Web界面**: Streamlit
- **HTTP客户端**: httpx
- **环境管理**: python-dotenv

## 📦 快速开始

### 1. 系统要求

- Python 3.12+
- MongoDB (本地或Atlas云服务)
- OpenAI API Key

### 2. 安装

```bash
# 克隆项目
git clone <repository-url>
cd CreatPartner

# 运行自动安装脚本
python install.py
```

### 3. 配置环境

编辑 `.env` 文件，添加必要的配置：

```bash
# 必需配置
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=mongodb://localhost:27017

# 可选配置
JINA_API_KEY=your_jina_api_key_here
DEFAULT_MODEL=openai:gpt-4o
DB_NAME=creatpartner
MAX_SEARCH_RESULTS=5
```

### 4. 启动系统

```bash
# 基础演示
python main.py

# 完整异步演示
python main.py --async

# Streamlit Web界面 (开发中)
streamlit run app.py
```

## 💻 使用示例

### 基础搜索

```python
from search_agent import SearchAgent, SearchDependencies

# 创建搜索代理
agent = SearchAgent()
deps = SearchDependencies(max_results=5)

# 执行搜索
result = agent.search_sync("AI在教育中的应用", deps)
print(result)
```

### 知识库管理

```python
from knowledge_agent import KnowledgeAgent, KnowledgeDependencies

# 创建知识库代理
agent = KnowledgeAgent()
deps = KnowledgeDependencies(
    mongodb_uri="mongodb://localhost:27017",
    database_name="my_project"
)

# 添加项目记忆
await agent.manage_project_memory(
    "add",
    "确定使用React + Node.js技术栈",
    deps
)

# 处理外部资料
external_data = [
    {"title": "React最佳实践", "content": "...", "source": "web"}
]
await agent.process_external_data(external_data, deps)
```

### 主代理使用

```python
from main_agent import create_creatpartner_agent

# 创建主代理
agent = create_creatpartner_agent()

# 开始研究会话
result = agent.start_research_session_sync(
    project_name="智能学习助手",
    project_description="基于AI的个性化教育平台"
)
print(result)
```

## 🏗️ 项目结构

```
CreatPartner/
├── search_agent.py         # 搜索代理实现
├── knowledge_agent.py      # 知识库代理实现
├── main_agent.py          # 主代理系统
├── main.py               # 演示程序
├── install.py            # 自动安装脚本
├── pyproject.toml        # 项目配置
├── .env.example          # 环境变量模板
├── .env.sample           # 示例配置
└── README.md            # 项目说明
```

## 🔧 配置说明

### 环境变量详解

| 变量名 | 说明 | 必需 | 默认值 |
|--------|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | 是 | - |
| `MONGODB_URI` | MongoDB连接字符串 | 是 | `mongodb://localhost:27017` |
| `JINA_API_KEY` | Jina搜索API密钥 | 否 | - |
| `DEFAULT_MODEL` | 默认AI模型 | 否 | `openai:gpt-4o` |
| `DB_NAME` | 数据库名称 | 否 | `creatpartner` |
| `MAX_SEARCH_RESULTS` | 最大搜索结果数 | 否 | `5` |
| `EMBEDDING_MODEL` | 嵌入模型 | 否 | `sentence-transformers/all-MiniLM-L6-v2` |

### MongoDB设置

1. **本地MongoDB**:
   ```bash
   # 启动MongoDB服务
   mongod --dbpath /path/to/data
   ```

2. **MongoDB Atlas**:
   - 创建Atlas集群
   - 获取连接字符串
   - 配置网络访问

## 🧪 开发指南

### 扩展搜索工具

```python
@search_agent.tool
async def custom_search(ctx: RunContext, query: str):
    # 实现自定义搜索逻辑
    pass
```

### 添加知识处理器

```python
@knowledge_agent.tool  
async def process_custom_data(ctx: RunContext, data: Dict):
    # 实现自定义数据处理逻辑
    pass
```

### 自定义主代理行为

```python
# 修改system_prompt来调整AI行为
agent = Agent(
    model_name,
    system_prompt="你的自定义提示..."
)
```

## 🔍 功能特性

### 智能搜索
- 多源信息整合
- 语义相似度匹配
- 实时结果分析
- 自动去重和排序

### 知识管理
- 向量化存储
- 语义检索
- 自动分类标记
- 版本控制

### 项目分析
- 技术可行性评估
- 市场竞争分析
- 创新点识别
- 风险评估

### 协作支持
- 多用户会话
- 知识共享
- 进度跟踪
- 决策记录

## 📊 使用场景

### 创新创业竞赛
- 项目立项调研
- 技术方案设计
- 市场分析报告
- 商业计划书

### 学术研究
- 文献调研
- 研究方向探索
- 论文写作支持
- 实验设计

### 产品开发
- 需求分析
- 技术选型
- 竞品分析
- 用户研究

## ⚠️ 注意事项

1. **API密钥安全**: 不要将API密钥提交到版本控制系统
2. **数据库备份**: 定期备份重要的项目知识库
3. **网络访问**: 确保能访问OpenAI、Jina等API服务
4. **资源限制**: 注意各API的调用频率和费用限制
5. **数据隐私**: 敏感项目信息建议使用本地部署

## 🤝 贡献指南

欢迎提交Issues和Pull Requests来改进这个项目！

### 开发流程
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

### 代码规范
- 遵循PEP 8
- 添加类型注解
- 编写测试用例
- 更新文档

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

- [Pydantic AI](https://ai.pydantic.org.cn/) - 强大的AI Agent框架
- [MongoDB](https://www.mongodb.com/) - 优秀的向量数据库
- [Jina](https://jina.ai/) - 先进的搜索技术
- [Sentence Transformers](https://www.sbert.net/) - 文本嵌入模型

---

**CreatPartner** - 让AI助力你的创新创业之路！ 🚀
