"""
CreatPartner 主代理系统
基于Pydantic AI多代理设计模式重新架构
实现智能体委托和程序化智能体交接
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# 导入配置
from config import config, get_model_name, create_llm_provider, validate_config

try:
    from pydantic_ai import Agent, RunContext, RunUsage, UsageLimits
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models.openai import OpenAIChatModel
    
    # 导入子代理
    from search_agent import SearchAgent, SearchDependencies
    from knowledge_agent import KnowledgeAgent, KnowledgeDependencies
    
except ImportError as e:
    print(f"警告: 缺少依赖包 {e}")
    # 模拟类定义
    class Agent:
        def __init__(self, *args, **kwargs):
            pass
        def tool(self, func):
            return func
        def run_sync(self, *args, **kwargs):
            return type('Result', (), {'output': '模拟结果 - 请安装依赖包'})()
    class RunContext:
        pass


@dataclass 
class SharedDependencies:
    """多代理共享依赖配置"""
    # API密钥
    jina_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # 数据库配置
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "creatpartner"
    
    # 项目配置
    project_name: str = "未命名项目"
    project_description: str = ""
    project_stage: Literal["planning", "research", "development", "testing", "deployment"] = "planning"
    
    # 搜索配置
    max_search_results: int = 5
    max_knowledge_results: int = 10
    
    # 使用限制
    usage_limits: Optional[UsageLimits] = None
    
    def __post_init__(self):
        """从配置文件初始化默认值"""
        if self.jina_api_key is None:
            self.jina_api_key = config.embedding.api_key
        if self.openai_api_key is None:
            self.openai_api_key = config.llm.api_key
        if self.mongodb_uri == "mongodb://localhost:27017":
            self.mongodb_uri = config.database.mongodb_uri
        if self.database_name == "creatpartner":
            self.database_name = config.database.database_name
        if self.max_search_results == 5:
            self.max_search_results = config.search.max_results


class ResearchCoordinator:
    """研究协调器 - 负责智能体委托和工作流协调"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_model_name()
        
        # 创建自定义模型实例
        model = self._create_model(model_name)
        
        # 主协调代理
        self.coordinator = Agent(
            model,
            deps_type=SharedDependencies,
            system_prompt="""
            你是CreatPartner的研究协调器，负责协调搜索代理和知识代理的工作。
            
            你的核心职责：
            1. 理解用户需求并制定研究策略
            2. 决定何时委托给搜索代理或知识代理
            3. 整合不同代理的结果
            4. 确保项目知识库的完整性和一致性
            
            工作原则：
            - 优先使用现有知识库，必要时才进行外部搜索
            - 确保搜索结果被适当地存储到知识库
            - 维护项目的长期记忆和上下文
            - 提供结构化和可操作的建议
            """,
        )
        
        # 子代理实例
        self.search_agent = SearchAgent(model_name)
        self.knowledge_agent = KnowledgeAgent(model_name)
        
        # 注册协调工具
        self._register_coordination_tools()
    
    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                # 使用自定义提供商
                provider = create_llm_provider()
                if provider:
                    return OpenAIChatModel(
                        config.llm.model_name,
                        provider=provider
                    )
            
            # 回退到默认模型
            return model_name
        except Exception as e:
            print(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name
    
    def _register_coordination_tools(self):
        """注册协调工具 - 实现智能体委托"""
        
        @self.coordinator.tool
        async def delegate_search_task(
            ctx: RunContext[SharedDependencies],
            query: str,
            search_type: Literal["web", "arxiv", "both"] = "both",
            store_results: bool = True
        ) -> Dict[str, Any]:
            """委托搜索任务给搜索代理
            
            Args:
                query: 搜索查询
                search_type: 搜索类型
                store_results: 是否存储搜索结果到知识库
                
            Returns:
                搜索结果和存储状态
            """
            try:
                # 创建搜索代理依赖
                search_deps = SearchDependencies(
                    jina_api_key=ctx.deps.jina_api_key,
                    max_results=ctx.deps.max_search_results
                )
                
                # 委托给搜索代理（智能体委托）
                if search_type == "web":
                    search_result = await self.search_agent.agent.run(
                        f"使用web_search工具搜索: {query}",
                        deps=search_deps,
                        usage=ctx.usage,  # 传递usage以合并统计
                        usage_limits=ctx.deps.usage_limits
                    )
                elif search_type == "arxiv":
                    search_result = await self.search_agent.agent.run(
                        f"使用arxiv_search工具搜索学术论文: {query}",
                        deps=search_deps,
                        usage=ctx.usage,
                        usage_limits=ctx.deps.usage_limits
                    )
                else:  # both
                    search_result = await self.search_agent.agent.run(
                        f"同时使用web_search和arxiv_search工具搜索: {query}",
                        deps=search_deps,
                        usage=ctx.usage,
                        usage_limits=ctx.deps.usage_limits
                    )
                
                result = {
                    "search_completed": True,
                    "search_type": search_type,
                    "results": search_result.output,
                    "stored": False
                }
                
                # 如果需要存储结果，委托给知识代理
                if store_results:
                    storage_result = await self.delegate_knowledge_storage(
                        ctx, query, search_result.output, "external_research"
                    )
                    result["stored"] = storage_result.get("success", False)
                    result["storage_details"] = storage_result
                
                return result
                
            except Exception as e:
                return {
                    "search_completed": False,
                    "error": str(e),
                    "message": f"搜索任务委托失败: {e}"
                }
        
        @self.coordinator.tool
        async def delegate_knowledge_query(
            ctx: RunContext[SharedDependencies],
            query: str,
            knowledge_type: Optional[Literal["project_memory", "external_research"]] = None,
            use_reranker: bool = True
        ) -> Dict[str, Any]:
            """委托知识查询给知识代理
            
            Args:
                query: 查询内容
                knowledge_type: 知识类型过滤
                use_reranker: 是否使用重排序
                
            Returns:
                查询结果
            """
            try:
                # 创建知识代理依赖
                knowledge_deps = KnowledgeDependencies(
                    mongodb_uri=ctx.deps.mongodb_uri,
                    database_name=ctx.deps.database_name,
                    jina_api_key=ctx.deps.jina_api_key,
                    max_results=ctx.deps.max_knowledge_results
                )
                
                # 委托给知识代理（智能体委托）
                knowledge_result = await self.knowledge_agent.agent.run(
                    f"搜索知识库: {query}" + 
                    (f" (类型: {knowledge_type})" if knowledge_type else "") +
                    (f" (使用重排序: {use_reranker})" if use_reranker else ""),
                    deps=knowledge_deps,
                    usage=ctx.usage,
                    usage_limits=ctx.deps.usage_limits
                )
                
                return {
                    "query_completed": True,
                    "query": query,
                    "knowledge_type": knowledge_type,
                    "results": knowledge_result.output
                }
                
            except Exception as e:
                return {
                    "query_completed": False,
                    "error": str(e),
                    "message": f"知识查询委托失败: {e}"
                }
        
        @self.coordinator.tool  
        async def delegate_knowledge_storage(
            ctx: RunContext[SharedDependencies],
            title: str,
            content: str,
            knowledge_type: Literal["project_memory", "external_research"],
            source: str = "coordinator",
            tags: List[str] = []
        ) -> Dict[str, Any]:
            """委托知识存储给知识代理
            
            Args:
                title: 知识标题
                content: 知识内容
                knowledge_type: 知识类型
                source: 来源
                tags: 标签
                
            Returns:
                存储结果
            """
            try:
                # 创建知识代理依赖
                knowledge_deps = KnowledgeDependencies(
                    mongodb_uri=ctx.deps.mongodb_uri,
                    database_name=ctx.deps.database_name,
                    jina_api_key=ctx.deps.jina_api_key
                )
                
                # 添加项目相关标签
                enhanced_tags = tags + [ctx.deps.project_name, ctx.deps.project_stage]
                
                # 委托给知识代理（智能体委托）
                storage_result = await self.knowledge_agent.agent.run(
                    f"添加知识到{knowledge_type}知识库: 标题='{title}', 内容='{content}', 来源='{source}', 标签={enhanced_tags}",
                    deps=knowledge_deps,
                    usage=ctx.usage,
                    usage_limits=ctx.deps.usage_limits
                )
                
                return {
                    "storage_completed": True,
                    "title": title,
                    "knowledge_type": knowledge_type,
                    "result": storage_result.output
                }
                
            except Exception as e:
                return {
                    "storage_completed": False,
                    "error": str(e),
                    "message": f"知识存储委托失败: {e}"
                }


class CreatPartnerAgent:
    """CreatPartner主代理 - 实现程序化智能体交接"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_model_name()
        
        # 研究协调器
        self.research_coordinator = ResearchCoordinator(model_name)
        
        # 创建自定义模型实例
        model = self._create_model(model_name)
        
        # 用户交互代理
        self.interaction_agent = Agent(
            model,
            deps_type=SharedDependencies,
            system_prompt="""
            你是CreatPartner，一个专为大学生创新创业竞赛设计的AI研究助手。
            
            你的角色：
            - 用户的主要交互接口
            - 理解和澄清用户需求
            - 制定研究和分析策略
            - 提供最终的建议和总结
            
            你有一个强大的研究协调器来帮助你：
            - delegate_search_task: 进行网络和学术搜索
            - delegate_knowledge_query: 查询项目知识库
            - delegate_knowledge_storage: 存储重要信息
            
            工作流程：
            1. 理解用户的具体需求
            2. 检查已有的项目知识
            3. 必要时进行外部搜索
            4. 整合信息并提供建议
            5. 记录重要决策和洞察
            
            特别关注：
            - 创新性和可行性分析
            - 市场和技术趋势
            - 竞品和合作机会
            - 风险识别和应对策略
            """,
        )
        
        # 为用户交互代理添加协调工具
        self._register_interaction_tools()
    
    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                # 使用自定义提供商
                provider = create_llm_provider()
                if provider:
                    return OpenAIChatModel(
                        config.llm.model_name,
                        provider=provider
                    )
            
            # 回退到默认模型
            return model_name
        except Exception as e:
            print(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name
    
    def _register_interaction_tools(self):
        """为用户交互代理注册协调工具"""
        
        @self.interaction_agent.tool
        async def delegate_search_task(
            ctx: RunContext[SharedDependencies],
            query: str,
            search_type: Literal["web", "arxiv", "both"] = "both",
            store_results: bool = True
        ) -> Dict[str, Any]:
            """委托搜索任务"""
            return await self.research_coordinator.delegate_search_task(
                ctx, query, search_type, store_results
            )
        
        @self.interaction_agent.tool
        async def delegate_knowledge_query(
            ctx: RunContext[SharedDependencies],
            query: str,
            knowledge_type: Optional[Literal["project_memory", "external_research"]] = None
        ) -> Dict[str, Any]:
            """委托知识查询"""
            return await self.research_coordinator.delegate_knowledge_query(
                ctx, query, knowledge_type, True
            )
        
        @self.interaction_agent.tool
        async def delegate_knowledge_storage(
            ctx: RunContext[SharedDependencies],
            title: str,
            content: str,
            knowledge_type: Literal["project_memory", "external_research"],
            tags: List[str] = []
        ) -> Dict[str, Any]:
            """委托知识存储"""
            return await self.research_coordinator.delegate_knowledge_storage(
                ctx, title, content, knowledge_type, "user_interaction", tags
            )
    
    async def research_workflow(
        self,
        user_query: str,
        deps: SharedDependencies,
        usage: Optional[RunUsage] = None
    ) -> str:
        """程序化智能体交接 - 研究工作流"""
        
        if usage is None:
            usage = RunUsage()
        
        # 第一步：检查现有知识
        print("🔍 第一步：检查项目知识库...")
        knowledge_result = await self.research_coordinator.coordinator.run(
            f"查询知识库: {user_query}",
            deps=deps,
            usage=usage,
            usage_limits=deps.usage_limits
        )
        
        # 第二步：决定是否需要外部搜索
        needs_search = self._analyze_knowledge_gap(knowledge_result.output, user_query)
        
        search_result = None
        if needs_search:
            print("🌐 第二步：进行外部搜索...")
            search_result = await self.research_coordinator.coordinator.run(
                f"执行搜索任务: {user_query} (类型: both, 存储: True)",
                deps=deps,
                usage=usage,
                usage_limits=deps.usage_limits
            )
        
        # 第三步：用户交互代理整合信息并提供最终回答
        print("🤖 第三步：整合信息并生成回答...")
        final_result = await self.interaction_agent.run(
            f"""
            用户查询: {user_query}
            
            现有知识库信息:
            {knowledge_result.output}
            
            外部搜索结果:
            {search_result.output if search_result else "无需外部搜索"}
            
            请基于以上信息为用户提供全面的回答和建议。
            """,
            deps=deps,
            usage=usage,
            usage_limits=deps.usage_limits
        )
        
        return final_result.output
    
    def _analyze_knowledge_gap(self, knowledge_result: str, query: str) -> bool:
        """分析知识库是否有信息缺口，决定是否需要外部搜索"""
        # 简单的规则：如果知识库结果为空或很少，则需要外部搜索
        if not knowledge_result or len(knowledge_result.strip()) < 100:
            return True
        
        # 如果结果中包含"未找到"、"没有相关"等关键词，需要外部搜索
        no_result_keywords = ["未找到", "没有相关", "无相关", "缺少信息"]
        return any(keyword in knowledge_result for keyword in no_result_keywords)
    
    async def start_project_session(
        self,
        project_name: str,
        project_description: str = "",
        project_stage: str = "planning",
        mongodb_uri: str = None,
        jina_api_key: str = None
    ) -> str:
        """开始项目会话 - 程序化智能体交接"""
        
        # 创建共享依赖
        deps = SharedDependencies(
            jina_api_key=jina_api_key or config.embedding.api_key,
            mongodb_uri=mongodb_uri or config.database.mongodb_uri,
            database_name=config.database.database_name,
            project_name=project_name,
            project_description=project_description,
            project_stage=project_stage,
            usage_limits=UsageLimits(
                request_limit=config.project.max_requests_per_session,
                total_tokens_limit=config.project.max_tokens_per_session,
                tool_calls_limit=config.project.max_tool_calls_per_session
            )
        )
        
        # 初始化知识库（如果需要）
        print("🗂️ 初始化项目知识库...")
        knowledge_deps = KnowledgeDependencies(
            mongodb_uri=deps.mongodb_uri,
            database_name=deps.database_name,
            jina_api_key=deps.jina_api_key
        )
        self.research_coordinator.knowledge_agent.create_vector_search_index(knowledge_deps)
        
        # 用户交互代理开始会话
        welcome_result = await self.interaction_agent.run(
            f"""
            项目会话开始：
            - 项目名称：{project_name}
            - 项目描述：{project_description}
            - 项目阶段：{project_stage}
            
            请向用户介绍你的能力，并询问他们希望如何开始研究。
            强调你可以：
            1. 搜索和整合外部信息
            2. 管理项目知识库
            3. 提供策略建议和分析
            4. 记录重要决策和洞察
            """,
            deps=deps
        )
        
        return welcome_result.output
    
    def start_project_session_sync(
        self,
        project_name: str,
        project_description: str = "",
        project_stage: str = "planning",
        mongodb_uri: str = None,
        jina_api_key: str = None
    ) -> str:
        """同步开始项目会话"""
        return asyncio.run(self.start_project_session(
            project_name, project_description, project_stage, mongodb_uri, jina_api_key
        ))
    
    async def chat(
        self,
        message: str,
        deps: SharedDependencies,
        message_history: Optional[list] = None
    ) -> str:
        """聊天接口 - 支持持续对话"""
        
        result = await self.interaction_agent.run(
            message,
            deps=deps,
            message_history=message_history,
            usage_limits=deps.usage_limits
        )
        
        return result.output


# 工厂函数
def create_creatpartner_agent(model_name: str = None) -> CreatPartnerAgent:
    """创建CreatPartner主代理实例"""
    return CreatPartnerAgent(model_name)


# 便利函数：创建标准依赖配置
def create_shared_dependencies(
    project_name: str,
    project_description: str = "",
    project_stage: str = "planning",
    **kwargs
) -> SharedDependencies:
    """创建共享依赖配置"""
    return SharedDependencies(
        jina_api_key=kwargs.get("jina_api_key") or config.embedding.api_key,
        openai_api_key=kwargs.get("openai_api_key") or config.llm.api_key,
        mongodb_uri=kwargs.get("mongodb_uri") or config.database.mongodb_uri,
        database_name=kwargs.get("database_name") or config.database.database_name,
        project_name=project_name,
        project_description=project_description,
        project_stage=project_stage,
        max_search_results=kwargs.get("max_search_results", config.search.max_results),
        max_knowledge_results=kwargs.get("max_knowledge_results", 10),
        usage_limits=UsageLimits(
            request_limit=kwargs.get("request_limit", config.project.max_requests_per_session),
            total_tokens_limit=kwargs.get("total_tokens_limit", config.project.max_tokens_per_session),
            tool_calls_limit=kwargs.get("tool_calls_limit", config.project.max_tool_calls_per_session)
        )
    )


# 使用示例
async def main():
    """多代理系统使用示例"""
    print("🚀 CreatPartner 多代理系统演示")
    print("=" * 50)
    
    # 创建主代理
    agent = create_creatpartner_agent()
    
    # 开始项目会话
    print("\n📋 开始项目会话...")
    welcome = await agent.start_project_session(
        project_name="智能环保监测系统",
        project_description="基于IoT和AI技术的智能环境监测平台",
        project_stage="research"
    )
    print(f"🤖 CreatPartner: {welcome}")
    
    # 创建依赖配置
    deps = create_shared_dependencies(
        project_name="智能环保监测系统",
        project_description="基于IoT和AI技术的智能环境监测平台",
        project_stage="research"
    )
    
    # 演示程序化智能体交接工作流
    print("\n🔄 演示研究工作流...")
    research_query = "当前环保监测技术的最新发展趋势和我们项目的技术可行性分析"
    
    result = await agent.research_workflow(research_query, deps)
    print(f"\n📊 研究结果:\n{result}")
    
    # 演示持续对话
    print("\n💬 演示持续对话...")
    follow_up = await agent.chat(
        "基于刚才的分析，我们应该重点关注哪些技术方向？",
        deps
    )
    print(f"\n🤖 CreatPartner: {follow_up}")


if __name__ == "__main__":
    asyncio.run(main())
