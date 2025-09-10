import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from config import config, get_model_name, create_llm_provider
from logger import warning

try:
    import httpx
    import pymongo
    from pymongo import MongoClient
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext, ModelRetry
    from pydantic_ai.models.openai import OpenAIChatModel

except ImportError as e:
    warning(f"缺少依赖包 {e}. 请运行: uv add pymongo httpx pydantic-ai python-dotenv")

    # 创建模拟类
    class BaseModel:
        pass

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, func):
            return func

        def instructions(self, func):
            return func

        def run_sync(self, *args, **kwargs):
            return type("Result", (), {"output": "模拟结果 - 请安装依赖包"})()

    class RunContext:
        pass

    class ModelRetry:
        pass


class KnowledgeType(Enum):
    """知识类型枚举"""

    PROJECT_MEMORY = "project_memory"  # 项目长期记忆
    EXTERNAL_RESEARCH = "external_research"  # 外部检索资料


class KnowledgeEntry(BaseModel):
    """知识条目模型"""

    id: Optional[str] = None
    title: str
    content: str
    knowledge_type: KnowledgeType
    source: str  # 来源：如"user_input", "web_search", "arxiv"
    tags: List[str] = []
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}


class KnowledgeAddResult(BaseModel):
    """知识添加结果的结构化输出"""

    success: bool
    ids: List[str] = []
    chunks_count: int = 0
    message: str
    error: Optional[str] = None


class KnowledgeSearchResult(BaseModel):
    """知识搜索结果的结构化输出"""

    success: bool
    results: List[Dict[str, Any]] = []
    total: int = 0
    reranked: bool = False
    message: str
    error: Optional[str] = None


class KnowledgeUpdateResult(BaseModel):
    """知识更新结果的结构化输出"""

    success: bool
    message: str
    collection: Optional[str] = None
    error: Optional[str] = None


class KnowledgeStatsResult(BaseModel):
    """知识库统计结果的结构化输出"""

    success: bool
    stats: Dict[str, Any] = {}
    message: str
    error: Optional[str] = None


@dataclass
class KnowledgeDependencies:
    """知识库代理依赖"""

    mongodb_uri: str
    database_name: str = "creatpartner"
    jina_api_key: Optional[str] = None
    embedding_model: str = "jina-embeddings-v3"
    embedding_dimensions: int = 1024
    max_results: int = 5


class JinaEmbeddingService:
    """基于Jina AI的嵌入服务"""

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.jina.ai/v1"

    async def get_embeddings(
        self, texts: List[str], task: str = "retrieval.passage"
    ) -> List[List[float]]:
        """使用Jina API生成嵌入"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
            "task": task,
            "embedding_type": "float",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings", headers=headers, json=payload
                )
                response.raise_for_status()

                result = response.json()
                return [item["embedding"] for item in result["data"]]

        except Exception as e:
            print(f"Jina嵌入生成失败: {e}")
            # 返回默认维度的零向量
            return [[0.0] * config.embedding.dimensions for _ in texts]

    async def get_single_embedding(
        self, text: str, task: str = "retrieval.passage"
    ) -> List[float]:
        """生成单个文本的嵌入"""
        embeddings = await self.get_embeddings([text], task)
        return embeddings[0] if embeddings else [0.0] * config.embedding.dimensions


class JinaRerankerService:
    """基于Jina AI的重排序服务"""

    def __init__(self, api_key: str, model: str = "jina-reranker-v2-base-multilingual"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.jina.ai/v1"

    async def rerank_documents(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """使用Jina Reranker重新排序文档"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": True,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/rerank", headers=headers, json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result["results"]

        except Exception as e:
            print(f"Jina重排序失败: {e}")
            # 返回原始顺序
            return [
                {"index": i, "document": doc, "relevance_score": 0.5}
                for i, doc in enumerate(documents)
            ]

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """重排序方法的别名，兼容旧代码"""
        return await self.rerank_documents(query, documents, top_n)


class JinaSegmenterService:
    """基于Jina AI的文本分割服务"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://segment.jina.ai"

    async def segment_text(
        self, content: str, max_chunk_length: int = 1000, tokenizer: str = "cl100k_base"
    ) -> List[str]:
        """使用Jina Segmenter分割文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "content": content,
            "tokenizer": tokenizer,
            "return_chunks": True,
            "max_chunk_length": max_chunk_length,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url, headers=headers, json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result.get("chunks", [content])

        except Exception as e:
            print(f"Jina文本分割失败: {e}")
            # 简单分割作为备选方案
            return [
                content[i : i + max_chunk_length]
                for i in range(0, len(content), max_chunk_length)
            ]


class KnowledgeAgent:
    """知识库代理 - 基于Jina AI和MongoDB的RAG实现"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_model_name()

        # 创建自定义模型实例
        model = self._create_model(model_name)

        self.agent = Agent(
            model,
            deps_type=KnowledgeDependencies,
            instructions="""
            你是一个专业的知识库管理助手，负责管理大学生创新创业项目的知识体系。
            
            你的核心职责：
            1. 管理项目长期记忆知识库 - 存储项目历史、决策、经验教训
            2. 管理外部检索资料知识库 - 整理和归档从外部获取的研究资料
            3. 提供智能的知识检索和关联分析
            4. 协助知识库的维护和优化
            
            工作原则：
            - 确保知识的准确性和可追溯性
            - 建立有效的知识分类和标签体系
            - 提供上下文相关的知识推荐
            - 支持知识的版本管理和更新
            
            特别关注：
            - 项目的创新点和核心价值
            - 技术方案的可行性分析
            - 市场调研和竞争分析
            - 风险评估和应对策略
            """,
            # 设置重试机制
            retries=2,
        )

        # 初始化组件
        self.client = None
        self.jina_embedding = None
        self.jina_reranker = None
        self.jina_segmenter = None
        self._register_tools()

    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                # 使用自定义提供商
                provider = create_llm_provider()
                if provider:
                    return OpenAIChatModel(config.llm.model_name, provider=provider)

            # 回退到默认模型
            return model_name
        except Exception as e:
            print(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name

    def _initialize_components(self, deps: KnowledgeDependencies):
        """初始化MongoDB连接和Jina服务"""
        if self.client is None:
            self.client = MongoClient(deps.mongodb_uri)

        if self.jina_embedding is None and deps.jina_api_key:
            self.jina_embedding = JinaEmbeddingService(
                deps.jina_api_key, deps.embedding_model
            )

        if self.jina_reranker is None and deps.jina_api_key:
            self.jina_reranker = JinaRerankerService(deps.jina_api_key)

        if self.jina_segmenter is None and deps.jina_api_key:
            self.jina_segmenter = JinaSegmenterService(deps.jina_api_key)

    async def _get_embedding(self, text: str) -> List[float]:
        """生成文本嵌入"""
        if self.jina_embedding:
            return await self.jina_embedding.get_single_embedding(text)
        else:
            # 返回默认维度的零向量
            return [0.0] * config.embedding.dimensions

    def _register_tools(self):
        """注册知识库管理工具"""

        @self.agent.tool
        async def add_knowledge(
            ctx: RunContext[KnowledgeDependencies],
            title: str,
            content: str,
            knowledge_type: str,
            source: str,
            tags: List[str] = [],
        ) -> KnowledgeAddResult:
            """添加知识条目到知识库

            Args:
                title: 知识标题
                content: 知识内容
                knowledge_type: 知识类型 (project_memory 或 external_research)
                source: 来源标识
                tags: 标签列表

            Returns:
                添加结果信息
            """
            try:
                self._initialize_components(ctx.deps)

                # 验证知识类型
                if knowledge_type not in ["project_memory", "external_research"]:
                    return KnowledgeAddResult(
                        success=False,
                        message=f"无效的知识类型: {knowledge_type}",
                        error="知识类型必须是 'project_memory' 或 'external_research'",
                    )

                # 使用Jina分割长文本
                chunks = [content]
                if self.jina_segmenter and len(content) > config.llm.chunk_size:
                    try:
                        chunks = await self.jina_segmenter.segment_text(
                            content, max_chunk_length=config.llm.chunk_size
                        )
                    except Exception as e:
                        if config.project.debug_mode:
                            print(f"文本分割失败，使用原文本: {e}")

                db = self.client[ctx.deps.database_name]
                collection = db[f"knowledge_{knowledge_type}"]

                # 为每个chunk创建知识条目
                inserted_ids = []
                for i, chunk in enumerate(chunks):
                    try:
                        # 生成嵌入
                        embedding = await self._get_embedding(f"{title} {chunk}")

                        entry = KnowledgeEntry(
                            title=f"{title} (部分 {i + 1})"
                            if len(chunks) > 1
                            else title,
                            content=chunk,
                            knowledge_type=KnowledgeType(knowledge_type),
                            source=source,
                            tags=tags,
                            embedding=embedding,
                        )

                        # 插入文档
                        doc = entry.dict()
                        doc["created_at"] = doc["created_at"].isoformat()
                        doc["updated_at"] = doc["updated_at"].isoformat()
                        doc["knowledge_type"] = doc["knowledge_type"].value

                        result = collection.insert_one(doc)
                        inserted_ids.append(str(result.inserted_id))

                    except Exception as chunk_error:
                        if config.project.debug_mode:
                            print(f"插入chunk {i + 1} 失败: {chunk_error}")
                        continue

                if inserted_ids:
                    return KnowledgeAddResult(
                        success=True,
                        ids=inserted_ids,
                        chunks_count=len(chunks),
                        message=f"成功添加{knowledge_type}知识条目: {title} ({len(inserted_ids)}个片段)",
                    )
                else:
                    return KnowledgeAddResult(
                        success=False,
                        message="没有成功插入任何知识片段",
                        error="所有插入操作都失败了",
                    )

            except Exception as e:
                error_msg = str(e)
                if config.project.debug_mode:
                    print(f"添加知识条目失败: {error_msg}")

                return KnowledgeAddResult(
                    success=False,
                    message=f"添加知识条目失败: {error_msg}",
                    error=error_msg,
                )

        @self.agent.tool
        async def search_knowledge(
            ctx: RunContext[KnowledgeDependencies],
            query: str,
            knowledge_type: Optional[str] = None,
            limit: int = 5,
            use_reranker: bool = False,  # 默认禁用重排序以避免额外的API调用
        ) -> KnowledgeSearchResult:
            """搜索知识库

            Args:
                query: 搜索查询
                knowledge_type: 知识类型过滤 (project_memory 或 external_research)
                limit: 返回结果数量限制
                use_reranker: 是否使用重排序优化结果

            Returns:
                搜索结果
            """
            try:
                if config.project.debug_mode:
                    print(f"🔍 搜索知识库: {query}")

                self._initialize_components(ctx.deps)

                if not self.client:
                    return KnowledgeSearchResult(
                        success=False,
                        message="数据库连接未初始化",
                        error="MongoDB客户端未初始化",
                    )

                # 生成查询嵌入
                query_embedding = await self._get_embedding(query)

                db = self.client[ctx.deps.database_name]
                results = []

                # 确定搜索的集合
                collections_to_search = []
                if knowledge_type and knowledge_type in [
                    "project_memory",
                    "external_research",
                ]:
                    collections_to_search = [f"knowledge_{knowledge_type}"]
                else:
                    collections_to_search = [
                        "knowledge_project_memory",
                        "knowledge_external_research",
                    ]

                if config.project.debug_mode:
                    print(f"   搜索集合: {collections_to_search}")

                # 在每个集合中搜索
                for collection_name in collections_to_search:
                    try:
                        collection = db[collection_name]

                        # 首先检查集合是否存在以及是否有数据
                        doc_count = collection.count_documents({})
                        if config.project.debug_mode:
                            print(f"   集合 {collection_name} 文档数: {doc_count}")

                        if doc_count == 0:
                            # 如果集合为空，创建一个测试文档
                            test_doc = {
                                "title": "测试知识条目",
                                "content": "这是一个测试知识条目，用于验证搜索功能",
                                "knowledge_type": knowledge_type or "project_memory",
                                "source": "system_test",
                                "tags": ["测试", "系统"],
                                "created_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat(),
                                "embedding": [0.1]
                                * config.embedding.dimensions,  # 简单的测试嵌入
                            }
                            collection.insert_one(test_doc)
                            if config.project.debug_mode:
                                print(f"   创建测试文档到 {collection_name}")

                        # 尝试向量搜索（但由于可能没有索引，预期会失败）
                        vector_search_success = False
                        try:
                            if query_embedding and len(query_embedding) > 0:
                                pipeline = [
                                    {
                                        "$vectorSearch": {
                                            "index": "vector_index",
                                            "path": "embedding",
                                            "queryVector": query_embedding,
                                            "numCandidates": 50,
                                            "limit": limit * 2
                                            if use_reranker
                                            else limit,
                                        }
                                    },
                                    {
                                        "$addFields": {
                                            "score": {"$meta": "vectorSearchScore"}
                                        }
                                    },
                                ]

                                cursor = collection.aggregate(pipeline)
                                vector_results = list(cursor)

                                for doc in vector_results:
                                    doc["_id"] = str(doc["_id"])
                                    doc["collection"] = collection_name
                                    doc["search_type"] = "vector"
                                    results.append(doc)

                                vector_search_success = True
                                if config.project.debug_mode:
                                    print(
                                        f"   向量搜索成功，找到 {len(vector_results)} 个结果"
                                    )

                        except Exception as vector_error:
                            if config.project.debug_mode:
                                print(f"   向量搜索失败: {vector_error}")

                        # 如果向量搜索失败或没有结果，使用文本搜索
                        if not vector_search_success or len(results) == 0:
                            if config.project.debug_mode:
                                print(f"   使用文本搜索作为备选方案")

                            try:
                                # 构建更宽泛的搜索条件
                                search_conditions = []

                                # 简单的文本匹配
                                if query:
                                    search_conditions.extend(
                                        [
                                            {
                                                "title": {
                                                    "$regex": query,
                                                    "$options": "i",
                                                }
                                            },
                                            {
                                                "content": {
                                                    "$regex": query,
                                                    "$options": "i",
                                                }
                                            },
                                        ]
                                    )

                                    # 搜索包含查询词的标签
                                    search_conditions.append(
                                        {"tags": {"$regex": query, "$options": "i"}}
                                    )

                                # 如果没有特定搜索条件，返回最近的文档
                                if not search_conditions:
                                    search_query = {}
                                else:
                                    search_query = {"$or": search_conditions}

                                text_results = collection.find(search_query).limit(
                                    limit
                                )

                                text_count = 0
                                for doc in text_results:
                                    doc["_id"] = str(doc["_id"])
                                    doc["score"] = 0.8  # 文本匹配的默认分数
                                    doc["collection"] = collection_name
                                    doc["search_type"] = "text"
                                    results.append(doc)
                                    text_count += 1

                                if config.project.debug_mode:
                                    print(f"   文本搜索找到 {text_count} 个结果")

                            except Exception as text_error:
                                if config.project.debug_mode:
                                    print(f"   文本搜索也失败: {text_error}")

                                # 最后的备选方案：返回任意文档
                                try:
                                    fallback_results = collection.find({}).limit(limit)
                                    fallback_count = 0
                                    for doc in fallback_results:
                                        doc["_id"] = str(doc["_id"])
                                        doc["score"] = 0.5
                                        doc["collection"] = collection_name
                                        doc["search_type"] = "fallback"
                                        results.append(doc)
                                        fallback_count += 1

                                    if config.project.debug_mode:
                                        print(
                                            f"   备选搜索返回 {fallback_count} 个结果"
                                        )

                                except Exception as fallback_error:
                                    if config.project.debug_mode:
                                        print(f"   备选搜索失败: {fallback_error}")

                    except Exception as collection_error:
                        if config.project.debug_mode:
                            print(
                                f"搜索集合 {collection_name} 完全失败: {collection_error}"
                            )
                        continue

                # 按分数排序
                if results:
                    results.sort(key=lambda x: x.get("score", 0), reverse=True)
                    results = results[:limit]  # 限制结果数量

                # 使用Jina重排序优化结果（可选）
                if use_reranker and self.jina_reranker and len(results) > 1:
                    try:
                        documents = [
                            f"{doc.get('title', '')}: {doc.get('content', '')}"
                            for doc in results
                        ]
                        reranked_results = await self.jina_reranker.rerank(
                            query, documents, top_n=limit
                        )

                        # 重新排序结果
                        reordered_results = []
                        for item in reranked_results:
                            if item.get("index", -1) < len(results):
                                result = results[item["index"]].copy()
                                result["rerank_score"] = item.get("relevance_score", 0)
                                reordered_results.append(result)

                        results = reordered_results
                        if config.project.debug_mode:
                            print(f"   重排序完成，最终结果数: {len(results)}")

                    except Exception as rerank_error:
                        if config.project.debug_mode:
                            print(f"重排序失败，使用原始结果: {rerank_error}")

                success_message = f"找到 {len(results)} 条相关知识"
                if not results:
                    success_message = "未找到相关知识，可能需要先添加一些内容到知识库"

                if config.project.debug_mode:
                    print(f"✅ {success_message}")

                return KnowledgeSearchResult(
                    success=True,
                    results=results,
                    total=len(results),
                    reranked=use_reranker
                    and self.jina_reranker is not None
                    and len(results) > 1,
                    message=success_message,
                )

            except Exception as e:
                error_msg = str(e)
                if config.project.debug_mode:
                    print(f"❌ 搜索知识库失败: {error_msg}")

                return KnowledgeSearchResult(
                    success=False,
                    message=f"搜索知识库失败: {error_msg}",
                    error=error_msg,
                )

        @self.agent.tool
        async def update_knowledge(
            ctx: RunContext[KnowledgeDependencies],
            knowledge_id: str,
            title: Optional[str] = None,
            content: Optional[str] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """更新知识条目

            Args:
                knowledge_id: 知识条目ID
                title: 新标题（可选）
                content: 新内容（可选）
                tags: 新标签（可选）

            Returns:
                更新结果
            """
            try:
                self._initialize_components(ctx.deps)

                # 构建更新字段
                update_fields = {"updated_at": datetime.now().isoformat()}
                if title:
                    update_fields["title"] = title
                if content:
                    update_fields["content"] = content
                if tags is not None:
                    update_fields["tags"] = tags

                # 如果标题或内容有更新，重新生成嵌入
                if title or content:
                    new_text = f"{title or ''} {content or ''}"
                    update_fields["embedding"] = await self._get_embedding(new_text)

                db = self.client[ctx.deps.database_name]

                # 在两个集合中尝试更新
                from bson import ObjectId

                for collection_name in [
                    "knowledge_project_memory",
                    "knowledge_external_research",
                ]:
                    collection = db[collection_name]
                    result = collection.update_one(
                        {"_id": ObjectId(knowledge_id)}, {"$set": update_fields}
                    )

                    if result.modified_count > 0:
                        return {
                            "success": True,
                            "message": f"成功更新知识条目 {knowledge_id}",
                            "collection": collection_name,
                        }

                return {"success": False, "message": f"未找到知识条目 {knowledge_id}"}

            except Exception as e:
                return {"success": False, "error": str(e), "message": f"更新失败: {e}"}

        @self.agent.tool
        async def get_knowledge_stats(
            ctx: RunContext[KnowledgeDependencies],
        ) -> Dict[str, Any]:
            """获取知识库统计信息

            Returns:
                知识库统计数据
            """
            try:
                self._initialize_components(ctx.deps)

                db = self.client[ctx.deps.database_name]

                stats = {
                    "project_memory": {
                        "total_entries": 0,
                        "sources": {},
                        "top_tags": [],
                    },
                    "external_research": {
                        "total_entries": 0,
                        "sources": {},
                        "top_tags": [],
                    },
                }

                # 统计项目记忆知识库
                project_collection = db["knowledge_project_memory"]
                stats["project_memory"]["total_entries"] = (
                    project_collection.count_documents({})
                )

                # 统计外部资料知识库
                research_collection = db["knowledge_external_research"]
                stats["external_research"]["total_entries"] = (
                    research_collection.count_documents({})
                )

                return {
                    "success": True,
                    "stats": stats,
                    "message": "成功获取知识库统计信息",
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"获取统计信息失败: {e}",
                }

    async def process_external_data(
        self, data: List[Dict[str, Any]], deps: Optional[KnowledgeDependencies] = None
    ) -> str:
        """处理外部检索到的数据并存入知识库"""
        if deps is None:
            deps = KnowledgeDependencies(
                mongodb_uri=config.database.mongodb_uri,
                database_name=config.database.database_name,
            )

        prompt = f"""
        请分析以下外部检索到的资料，并将其整理为结构化的知识条目。
        
        检索数据：
        {data}
        
        请为每个有价值的信息创建知识条目，包括：
        1. 合适的标题
        2. 内容摘要
        3. 相关标签
        4. 来源标识
        
        重点关注与创新创业项目相关的信息。
        """

        result = await self.agent.run(prompt, deps=deps)
        return result.output

    async def manage_project_memory(
        self, action: str, content: str, deps: Optional[KnowledgeDependencies] = None
    ) -> str:
        """管理项目长期记忆"""
        if deps is None:
            deps = KnowledgeDependencies(
                mongodb_uri=config.database.mongodb_uri,
                database_name=config.database.database_name,
            )

        prompt = f"""
        请执行以下项目记忆管理任务：
        
        操作类型：{action}
        内容：{content}
        
        根据操作类型，执行相应的知识库操作：
        - add: 添加新的项目记忆
        - search: 搜索相关项目记忆
        - update: 更新现有记忆
        - analyze: 分析项目知识库
        
        确保维护项目知识的完整性和可追溯性。
        """

        result = await self.agent.run(prompt, deps=deps)
        return result.output

    def create_vector_search_index(self, deps: KnowledgeDependencies):
        """创建向量搜索索引"""
        try:
            self._initialize_components(deps)

            db = self.client[deps.database_name]

            # 为两个知识库集合创建向量索引
            collections = ["knowledge_project_memory", "knowledge_external_research"]

            for collection_name in collections:
                collection = db[collection_name]

                # 创建向量搜索索引
                search_index_model = {
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "numDimensions": config.embedding.dimensions,  # 使用配置的嵌入维度
                                "path": "embedding",
                                "similarity": "cosine",
                            }
                        ]
                    },
                    "name": "vector_index",
                    "type": "vectorSearch",
                }

                try:
                    collection.create_search_index(search_index_model)
                    print(f"✅ 为 {collection_name} 创建向量索引成功")
                except Exception as e:
                    print(f"⚠️ 创建向量索引失败或已存在: {e}")

            # 创建文本索引
            for collection_name in collections:
                collection = db[collection_name]
                try:
                    collection.create_index(
                        [("title", "text"), ("content", "text"), ("tags", "text")]
                    )
                    print(f"✅ 为 {collection_name} 创建文本索引成功")
                except Exception as e:
                    print(f"⚠️ 创建文本索引失败或已存在: {e}")

        except Exception as e:
            print(f"❌ 创建索引失败: {e}")


# 工厂函数
def create_knowledge_agent(
    model_name: str = None, mongodb_uri: str = None
) -> KnowledgeAgent:
    """创建知识库代理实例"""
    return KnowledgeAgent(model_name)


# 使用示例
async def main():
    """示例用法"""
    # 创建知识库代理
    agent = create_knowledge_agent()

    # 配置依赖
    deps = KnowledgeDependencies(
        mongodb_uri=config.database.mongodb_uri, database_name="creatpartner_test"
    )

    # 创建索引
    agent.create_vector_search_index(deps)

    # 测试外部数据处理
    external_data = [
        {
            "title": "AI在教育中的应用趋势",
            "content": "人工智能技术在教育领域的应用...",
            "source": "web_search",
        },
        {
            "title": "创新创业项目评估标准",
            "content": "大学生创新创业项目的评估维度...",
            "source": "arxiv",
        },
    ]

    result = await agent.process_external_data(external_data, deps)
    print("外部数据处理结果:")
    print(result)

    # 测试项目记忆管理
    memory_result = await agent.manage_project_memory(
        "add", "项目技术栈确定：使用Python + MongoDB + Streamlit开发AI助手", deps
    )
    print("\n项目记忆管理结果:")
    print(memory_result)


if __name__ == "__main__":
    asyncio.run(main())
