import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal, Union
from enum import Enum

# 导入配置和组件
from config import config, get_model_name, create_llm_provider

# 导入日志系统
from logger import (
    get_logger,
    debug,
    info,
    warning,
    error,
    success,
    step,
    performance,
    agent_operation,
)

from search_agent import SearchAgent, create_search_agent, create_search_dependencies
from knowledge_agent import (
    KnowledgeAgent,
    create_knowledge_agent,
    KnowledgeDependencies,
)
from bp_reviewer_agent import (
    BPReviewerAgent,
    create_bp_reviewer_agent,
    CompetitionGroup,
)

# 导入Logfire集成 - 已移除，使用logger替代

try:
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext, ModelRetry
    from pydantic_ai.models.openai import OpenAIChatModel

except ImportError as e:
    warning(f"缺少依赖包 {e}. 请运行: uv add pydantic-ai")

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


class AgentRole(Enum):
    """Agent角色类型"""

    MAIN_ASSISTANT = "main_assistant"  # 主助手 - 与用户直接交流
    SEARCH_SPECIALIST = "search_specialist"  # 搜索专家 - 自动进行资料检索
    KNOWLEDGE_MANAGER = "knowledge_manager"  # 知识管理者 - 管理知识库


class ProjectStage(Enum):
    """项目阶段"""

    PLANNING = "planning"  # 规划阶段
    RESEARCH = "research"  # 调研阶段
    DEVELOPMENT = "development"  # 开发阶段
    TESTING = "testing"  # 测试阶段
    DEPLOYMENT = "deployment"  # 部署阶段
    COMPETITION = "competition"  # 比赛阶段


class TaskPriority(Enum):
    """任务优先级"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ProjectContext(BaseModel):
    """项目上下文信息"""

    name: str = "未命名项目"
    description: str = ""
    stage: ProjectStage = ProjectStage.PLANNING
    keywords: List[str] = []
    team_members: List[str] = []
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class TaskItem(BaseModel):
    """任务条目"""

    id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_agent: AgentRole
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    dependencies: List[str] = []  # 依赖的任务ID
    results: Optional[Dict[str, Any]] = None


class UserIntentClassification(BaseModel):
    """用户意图分类结果"""

    execution_sequence: List[
        Literal[
            "search",
            "knowledge_add",
            "knowledge_search",
            "analysis",
            "planning",
            "review_bp",
            "general",
        ]
    ]
    search_query: str
    knowledge_type: Literal["project_memory", "external_research"] = "external_research"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    reasoning: str
    final_goal: str  # 用户的最终目标


@dataclass
class ClassifierDependencies:
    """分类器依赖"""

    project_context: ProjectContext
    debug_mode: bool = False


class AgentResponse(BaseModel):
    """Agent响应结构"""

    agent_role: AgentRole
    content: str
    task_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    suggested_next_actions: List[str] = []
    confidence: float = 1.0


class SystemSummary(BaseModel):
    """系统摘要信息"""

    active_tasks: int
    completed_tasks: int
    knowledge_entries: int
    recent_activities: List[str]
    recommendations: List[str]


@dataclass
class MainAgentDependencies:
    """主Agent依赖配置"""

    project_context: ProjectContext
    search_agent: SearchAgent
    knowledge_agent: KnowledgeAgent
    bp_reviewer_agent: BPReviewerAgent  # 新增BP评审Agent
    knowledge_deps: KnowledgeDependencies
    session_id: str = "default"
    max_concurrent_tasks: int = 3
    enable_auto_search: bool = True
    enable_knowledge_update: bool = True
    # 新增：控制递归和复杂度的参数
    max_workflow_depth: int = 3  # 最大工作流程深度
    max_execution_time: int = 300  # 最大执行时间（秒）
    max_steps_per_workflow: int = 5  # 每个工作流程最大步骤数
    # 新增：递归控制参数
    current_recursion_depth: int = 0  # 当前递归深度
    max_recursion_depth: int = 3  # 最大递归深度
    task_execution_stack: list = None  # 任务执行栈，用于检测循环

    def __post_init__(self):
        """初始化后处理"""
        if self.task_execution_stack is None:
            self.task_execution_stack = []


class IntentClassifierAgent:
    """意图分类Agent - 基于Pydantic AI最佳实践"""

    def __init__(self, model_name: str = None):
        """初始化分类Agent"""
        if model_name is None:
            model_name = get_model_name()

        # 创建自定义模型实例
        model = self._create_model(model_name)

        # 创建分类Agent - 不使用强制结构化输出，改为文本输出后解析
        self.agent = Agent[ClassifierDependencies, str](
            model,
            deps_type=ClassifierDependencies,
            output_type=str,  # 改为字符串输出
            instructions="""
            你是CreatPartner系统的意图分类专家。你的任务是分析用户输入，准确识别用户的意图并分类为顺序执行序列。

            任务类型说明：
            1. search - 搜索外部信息、资料、文献等
            2. knowledge_add - 添加、存储、保存信息到知识库
            3. knowledge_search - 从已有知识库中查找信息
            4. analysis - 分析、评估、总结现有信息
            5. planning - 制定计划、策略、方案
            6. review_bp - 评审或分析商业计划书（BP）
            6. general - 一般性对话、咨询、介绍

            执行序列规则：
            - 按照逻辑顺序排列任务，确保前置依赖得到满足
            - "查找xxx并存入知识库" = ["search", "knowledge_add"]
            - "搜索xxx并保存" = ["search", "knowledge_add"] 
            - "制定xxx计划" = ["search", "planning"] (需要先收集信息)
            - "分析xxx情况" = ["knowledge_search", "analysis"] 或 ["search", "analysis"]
            - "补充xxx知识" = ["search", "knowledge_add"]
            - "评审我的商业计划书" = ["review_bp"]
            
            重要原则：
            - 如果需要外部信息才能完成任务，必须先执行search
            - 如果要存储信息到知识库，必须先获得信息内容
            - 避免循环依赖和重复任务
            - 序列中每个任务只出现一次
            - 限制最多3个步骤，避免过长的思维链

            请严格按照以下JSON格式返回结果（不要添加代码块标记）：
            {
                "execution_sequence": ["按顺序执行的任务类型列表"],
                "search_query": "搜索关键词",
                "knowledge_type": "project_memory或external_research",
                "priority": "low/medium/high/urgent",
                "reasoning": "执行序列的逻辑依据",
                "final_goal": "用户的最终目标描述"
            }
            """,
            retries=1,
        )

    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                provider = create_llm_provider()
                if provider:
                    from pydantic_ai.models.openai import OpenAIChatModel

                    return OpenAIChatModel(config.llm.model_name, provider=provider)
            return model_name
        except Exception as e:
            if config.project.debug_mode:
                error(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name

    def _parse_json_output(self, output: str) -> UserIntentClassification:
        """解析LLM的JSON输出"""
        try:
            import json
            import re

            # 清理输出
            output = output.strip()

            # 移除可能的代码块标记
            if output.startswith("```json"):
                output = output[7:]
            if output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]

            # 尝试提取JSON部分
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                output = json_match.group(0)

            # 解析JSON
            data = json.loads(output)

            # 验证和标准化字段
            valid_tasks = [
                "search",
                "knowledge_add",
                "knowledge_search",
                "analysis",
                "planning",
                "review_bp",
                "general",
            ]

            # 处理执行序列
            execution_sequence = data.get("execution_sequence", [])
            if not isinstance(execution_sequence, list):
                execution_sequence = []

            # 验证执行序列的有效性并去重
            validated_sequence = []
            for task in execution_sequence:
                if task in valid_tasks and task not in validated_sequence:
                    validated_sequence.append(task)

            # 限制最多3个步骤，避免过长的思维链
            if not validated_sequence:
                validated_sequence = ["general"]
            elif len(validated_sequence) > 3:
                validated_sequence = validated_sequence[:3]
                if config.project.debug_mode:
                    warning("执行序列被截断至3步，避免过长思维链")

            search_query = data.get("search_query", "")
            if not search_query:
                search_query = "未指定查询"

            knowledge_type = data.get("knowledge_type", "external_research")
            if knowledge_type not in ["project_memory", "external_research"]:
                knowledge_type = "external_research"

            priority = data.get("priority", "medium")
            if priority not in ["low", "medium", "high", "urgent"]:
                priority = "medium"

            reasoning = data.get("reasoning", "基于LLM智能分析")
            final_goal = data.get("final_goal", search_query)

            return UserIntentClassification(
                execution_sequence=validated_sequence,
                search_query=search_query,
                knowledge_type=knowledge_type,
                priority=priority,
                reasoning=reasoning,
                final_goal=final_goal,
            )

        except Exception as e:
            if config.project.debug_mode:
                error("JSON解析失败", error=str(e))
                debug("原始输出", content=output)

            # 返回默认值
            return UserIntentClassification(
                execution_sequence=["general"],
                search_query="解析失败",
                knowledge_type="external_research",
                priority="medium",
                reasoning=f"JSON解析失败: {str(e)}",
                final_goal="解析失败",
            )

    async def classify_intent(
        self, user_goal: str, project_context: ProjectContext = None
    ) -> UserIntentClassification:
        """分类用户意图

        Args:
            user_goal: 用户目标描述
            project_context: 项目上下文信息

        Returns:
            分类结果
        """
        try:
            if project_context is None:
                project_context = ProjectContext()

            # 创建依赖
            deps = ClassifierDependencies(
                project_context=project_context, debug_mode=config.project.debug_mode
            )

            # 构建详细的分类提示
            classification_prompt = f"""
            请分析以下用户目标，根据项目上下文进行准确的意图分类：

            用户目标: "{user_goal}"
            
            项目信息:
            - 项目名称: {project_context.name}
            - 项目阶段: {project_context.stage.value}
            - 项目描述: {project_context.description}
            - 项目关键词: {", ".join(project_context.keywords)}

            请返回JSON格式的分类结果。
            """

            # 执行分类
            result = await self.agent.run(classification_prompt, deps=deps)

            # 解析JSON输出
            classification = self._parse_json_output(result.output)

            if config.project.debug_mode:
                success("Agent分类成功", sequence=classification.execution_sequence)

            return classification

        except Exception as e:
            if config.project.debug_mode:
                error("Agent分类失败", error=str(e))

            # 返回默认分类结果
            return UserIntentClassification(
                execution_sequence=["general"],
                search_query=user_goal,
                knowledge_type="external_research",
                priority="medium",
                reasoning=f"Agent分类失败，使用默认值: {str(e)}",
                final_goal=user_goal,
            )

    def classify_intent_sync(
        self, user_goal: str, project_context: ProjectContext = None
    ) -> UserIntentClassification:
        """同步版本的意图分类"""
        return asyncio.run(self.classify_intent(user_goal, project_context))


class MainAgent:
    """主代理 - 协调三个专门的助手Agent"""

    def __init__(self, model_name: str = None, project_context: ProjectContext = None):
        if config.project.debug_mode:
            agent_operation("CreatPartner主代理", "初始化", "开始")

        if model_name is None:
            model_name = get_model_name()

        # 创建自定义模型实例
        model = self._create_model(model_name)

        # 初始化项目上下文
        self.project_context = project_context or ProjectContext()

        # 创建意图分类Agent
        self.intent_classifier = IntentClassifierAgent(model_name)

        # 创建专门的助手代理
        self.search_agent = create_search_agent(model_name)
        self.knowledge_agent = create_knowledge_agent(model_name)
        self.bp_reviewer_agent = create_bp_reviewer_agent(model_name)  # 新增

        # 任务管理
        self.tasks = {}  # 任务字典
        self.task_counter = 0

        # 创建主协调代理
        self.agent = Agent(
            model,
            deps_type=MainAgentDependencies,
            instructions=f"""
            你是CreatPartner - 专为大学生创新创业竞赛设计的AI助手系统的主协调者。
            
            当前项目信息：
            - 项目名称：{self.project_context.name}
            - 项目阶段：{self.project_context.stage.value}
            - 项目描述：{self.project_context.description}
            
            你的核心职责：
            1. 理解用户需求并智能选择合适的工具来执行任务
            2. 协调三个专门的助手Agent：
               - 搜索专家：负责资料检索和信息收集
               - 知识管理者：负责知识库管理和信息整理
               - 主助手：负责与用户直接交流和综合分析
            
            3. 管理任务队列和工作流程，提供任务执行结果
            4. 提供项目进展总结和建议
            
            工具使用规则：
            - 使用 delegate_search_task 来执行搜索相关的任务
            - 使用 delegate_knowledge_task 来管理知识库
            - 使用 delegate_review_task 来评审商业计划书
            - 使用 execute_dynamic_workflow 来执行基于智能分类的动态工作流程
            - 使用 get_system_status 来获取系统状态
            - 优先使用 execute_dynamic_workflow 处理复杂的复合任务
            - 单一任务可以直接使用专门的delegate工具
            
            动态工作流程特性：
            - 基于智能分类agent自动识别用户意图
            - 根据任务类型动态构建最优执行步骤
            - 支持复合任务的自动分解和执行
            - 提供详细的执行报告和分析结果
            
            工作原则：
            - 始终以用户的创新创业项目成功为目标
            - 优先提供有实用价值的信息和建议
            - 根据用户需求选择最合适的工具
            - 对于复合需求（如搜索+存储），自动执行所有必要步骤
            - 避免重复执行相同的任务
            - 保持信息的准确性和时效性
            
            响应格式：
            - 使用结构化的输出格式
            - 明确指出执行的任务和结果
            - 提供具体可行的建议
            - 包含项目发展的战略建议
            """,
            retries=2,
        )

        # 注册工具
        self._register_tools()

        if config.project.debug_mode:
            agent_operation("CreatPartner主代理", "初始化", "完成")

    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                provider = create_llm_provider()
                if provider:
                    return OpenAIChatModel(config.llm.model_name, provider=provider)
            return model_name
        except Exception as e:
            error(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name

    def _generate_task_id(self) -> str:
        """生成任务ID"""
        self.task_counter += 1
        return f"task_{self.task_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def _build_dynamic_workflow_plan(
        self, classification: UserIntentClassification, user_goal: str, ctx
    ) -> Dict[str, Any]:
        """基于智能分类结果构建动态工作流程计划"""

        execution_sequence = classification.execution_sequence
        max_steps = ctx.deps.max_steps_per_workflow

        # 限制执行序列长度，避免过长的思维链
        if len(execution_sequence) > max_steps:
            execution_sequence = execution_sequence[:max_steps]
            if config.project.debug_mode:
                warning(f"执行序列被截断至 {max_steps} 步")

        workflow_steps = []
        search_query = classification.search_query
        knowledge_type = classification.knowledge_type
        priority = classification.priority
        final_goal = classification.final_goal

        # 简化工作流程构建逻辑，避免过度复杂
        for i, task_type in enumerate(execution_sequence):
            if task_type == "search":
                workflow_steps.append(
                    {
                        "step": f"step_{i + 1}_search",
                        "action": "delegate_search_task",
                        "params": {
                            "query": search_query[:200],  # 限制查询长度
                            "search_type": "comprehensive",
                            "priority": priority,
                        },
                        "description": f"搜索: {search_query[:50]}",
                        "sequence_index": i,
                    }
                )

            elif task_type == "knowledge_add":
                content = (
                    user_goal[:500] if len(user_goal) > 500 else user_goal
                )  # 限制内容长度
                workflow_steps.append(
                    {
                        "step": f"step_{i + 1}_knowledge_add",
                        "action": "delegate_knowledge_task",
                        "params": {
                            "action": "add",
                            "content": content,
                            "knowledge_type": knowledge_type or "external_research",
                            "priority": priority,
                            "query": search_query,  # 添加查询信息，用于生成标题
                            "title": f"搜索结果 - {search_query[:50]}",  # 添加默认标题
                        },
                        "description": f"添加知识: {final_goal[:50]}",
                        "sequence_index": i,
                    }
                )

            elif task_type == "knowledge_search":
                workflow_steps.append(
                    {
                        "step": f"step_{i + 1}_knowledge_search",
                        "action": "delegate_knowledge_task",
                        "params": {
                            "action": "search",
                            "content": search_query[:200],
                            "knowledge_type": knowledge_type,
                            "priority": priority,
                        },
                        "description": f"搜索知识库: {search_query[:50]}",
                        "sequence_index": i,
                    }
                )

            elif task_type == "general":
                workflow_steps.append(
                    {
                        "step": f"step_{i + 1}_general",
                        "action": "general_response",
                        "params": {"content": user_goal[:300], "priority": priority},
                        "description": f"常规回复: {user_goal[:50]}",
                        "sequence_index": i,
                    }
                )

            # 早期终止：如果步骤已经足够处理用户目标
            if i >= 2 and task_type in ["general", "knowledge_search"]:
                break

        return {
            "workflow_type": f"sequential_{len(workflow_steps)}_steps",
            "execution_sequence": execution_sequence[: len(workflow_steps)],
            "steps": workflow_steps,
            "total_steps": len(workflow_steps),
            "estimated_duration": min(len(workflow_steps) * 20, 120),  # 限制最大时间
            "priority": priority,
            "reasoning": classification.reasoning[:200],  # 限制推理长度
            "final_goal": final_goal[:100],  # 限制目标描述长度
        }

    async def _execute_workflow_steps(
        self, workflow_steps: List[Dict[str, Any]], ctx
    ) -> List[Dict[str, Any]]:
        """执行工作流程步骤"""

        execution_results = []
        previous_step_result = None  # 用于存储上一步的结果

        for i, step in enumerate(workflow_steps):
            try:
                if config.project.debug_mode:
                    info(
                        f"执行步骤 {i + 1}/{len(workflow_steps)}",
                        description=step["description"],
                    )

                action = step["action"]
                params = step["params"].copy()  # 复制参数，避免修改原始数据

                # 核心修复：如果当前是 knowledge_add 并且上一步是 search，则使用上一步的结果
                if (
                    action == "delegate_knowledge_task"
                    and params.get("action") in ["add", "store"]
                    and previous_step_result
                    and previous_step_result.get("status") == "completed"
                    and previous_step_result.get("result")
                ):  # 修复：直接检查result是否存在
                    search_output = previous_step_result["result"]

                    # 如果是字符串，直接使用；如果是字典，提取有用信息
                    if isinstance(search_output, str):
                        search_content = search_output
                    elif isinstance(search_output, dict):
                        # 尝试提取各种可能的内容字段
                        search_content = (
                            search_output.get("summary")
                            or search_output.get("content")
                            or str(search_output)
                        )
                    else:
                        search_content = str(search_output)

                    if config.project.debug_mode:
                        debug(
                            "使用上一步的搜索结果更新知识库内容",
                            content_length=len(search_content),
                        )

                    # 更新参数：使用搜索结果作为知识库内容
                    params["content"] = search_content
                    # 更新标题为更有意义的描述
                    if params.get("query"):
                        params["title"] = f"搜索结果 - {params['query'][:50]}"
                    else:
                        # 尝试从前一个步骤获取查询信息
                        prev_step = workflow_steps[i - 1] if i > 0 else None
                        if prev_step and prev_step.get("params", {}).get("query"):
                            query = prev_step["params"]["query"]
                            params["title"] = f"搜索结果 - {query[:50]}"

                # 根据动作类型执行相应的委派任务
                if action == "delegate_search_task":
                    result = await self._execute_search_step(ctx, params)
                elif action == "delegate_knowledge_task":
                    result = await self._execute_knowledge_step(ctx, params)
                else:
                    result = {"status": "skipped", "message": f"未知动作类型: {action}"}

                execution_results.append(
                    {
                        "step_index": i,
                        "step": step["step"],
                        "description": step["description"],
                        "action": action,
                        "params": params,
                        "result": result,
                        "status": result.get("status", "unknown"),
                        "execution_time": datetime.now().isoformat(),
                    }
                )

                # 保存当前步骤的结果，供下一步使用
                previous_step_result = result

            except Exception as e:
                error_result = {
                    "step_index": i,
                    "step": step["step"],
                    "description": step["description"],
                    "action": step["action"],
                    "params": step["params"],
                    "result": {"error": str(e)},
                    "status": "failed",
                    "execution_time": datetime.now().isoformat(),
                }
                execution_results.append(error_result)

                # 即使失败，也更新 previous_step_result
                previous_step_result = error_result

                if config.project.debug_mode:
                    error(f"步骤执行失败: {e}")

        return execution_results

    async def _execute_search_step(self, ctx, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行搜索步骤"""
        try:
            # 创建搜索任务
            task_id = self._generate_task_id()
            task = TaskItem(
                id=task_id,
                title=f"动态搜索: {params['query'][:50]}...",
                description=f"执行{params.get('search_type', 'comprehensive')}搜索",
                priority=TaskPriority(params.get("priority", "medium")),
                assigned_agent=AgentRole.SEARCH_SPECIALIST,
                status="in_progress",
            )
            self.tasks[task_id] = task

            # 执行搜索
            search_deps = create_search_dependencies()
            result = await ctx.deps.search_agent.search(
                params["query"], search_deps, params.get("search_type", "comprehensive")
            )

            # 更新任务状态
            task.status = "completed"
            task.results = {"search_result": result}
            task.updated_at = datetime.now()

            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "search_type": params.get("search_type", "comprehensive"),
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _execute_knowledge_step(
        self, ctx, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行知识管理步骤"""
        try:
            # 创建知识管理任务
            task_id = self._generate_task_id()
            task = TaskItem(
                id=task_id,
                title=f"动态知识管理: {params['action']}",
                description=f"执行知识库{params['action']}操作",
                priority=TaskPriority(params.get("priority", "medium")),
                assigned_agent=AgentRole.KNOWLEDGE_MANAGER,
                status="in_progress",
            )
            self.tasks[task_id] = task

            # 执行知识管理操作
            action = params["action"]
            content = params["content"]
            knowledge_type = params.get("knowledge_type")
            title = params.get("title", "用户输入")  # 获取标题，默认为"用户输入"

            if action == "add" or action == "store":
                # 添加或存储知识 - 使用动态标题
                add_prompt = f"请使用add_knowledge工具添加知识：\n标题：{title}\n内容：{content}\n类型：{knowledge_type or 'external_research'}\n来源：dynamic_workflow"

                agent_result = await ctx.deps.knowledge_agent.agent.run(
                    add_prompt, deps=ctx.deps.knowledge_deps
                )
                result = agent_result.output

            elif action == "search":
                # 搜索知识
                search_prompt = f"请使用search_knowledge工具搜索: {content}"
                if knowledge_type:
                    search_prompt += f"，知识类型：{knowledge_type}"

                agent_result = await ctx.deps.knowledge_agent.agent.run(
                    search_prompt, deps=ctx.deps.knowledge_deps
                )
                result = agent_result.output

            elif action == "analyze":
                # 分析知识
                result = await ctx.deps.knowledge_agent.manage_project_memory(
                    "analyze", content, ctx.deps.knowledge_deps
                )

            else:
                result = f"已执行{action}操作，内容：{content}"

            # 更新任务状态
            task.status = "completed"
            task.results = {"knowledge_result": result}
            task.updated_at = datetime.now()

            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "action": action,
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _generate_workflow_report(
        self,
        classification: UserIntentClassification,
        workflow_plan: Dict[str, Any],
        execution_results: List[Dict[str, Any]],
        user_goal: str,
    ) -> str:
        """生成工作流程执行报告"""

        successful_steps = [
            r for r in execution_results if r.get("status") == "completed"
        ]
        failed_steps = [r for r in execution_results if r.get("status") == "failed"]

        report = f"""
# 动态工作流程执行报告

## 用户目标
{user_goal}

## 智能分类结果
- **执行序列**: {" → ".join(classification.execution_sequence)}
- **最终目标**: {classification.final_goal}
- **任务优先级**: {classification.priority}
- **分类依据**: {classification.reasoning}

## 工作流程概览
- **工作流程类型**: {workflow_plan["workflow_type"]}
- **总步骤数**: {workflow_plan["total_steps"]}
- **成功步骤**: {len(successful_steps)}
- **失败步骤**: {len(failed_steps)}
- **执行效率**: {len(successful_steps) / workflow_plan["total_steps"] * 100:.1f}%

## 执行详情
"""

        for i, result in enumerate(execution_results):
            status_emoji = (
                "✅"
                if result["status"] == "completed"
                else "❌"
                if result["status"] == "failed"
                else "⏸️"
            )
            report += f"\n### 步骤 {i + 1}: {result['description']}\n"
            report += f"{status_emoji} **状态**: {result['status']}\n"

            if result["status"] == "completed":
                # 提取结果摘要
                result_data = result.get("result", {})
                if isinstance(result_data, dict):
                    if "result" in result_data:
                        content = str(result_data["result"])[:200]
                        report += f"**结果**: {content}{'...' if len(str(result_data['result'])) > 200 else ''}\n"
                else:
                    content = str(result_data)[:200]
                    report += f"**结果**: {content}{'...' if len(str(result_data)) > 200 else ''}\n"
            elif result["status"] == "failed":
                error = result.get("result", {}).get("error", "未知错误")
                report += f"**错误**: {error}\n"

        # 总结和建议
        report += f"\n## 总结\n"
        if len(successful_steps) == workflow_plan["total_steps"]:
            report += "🎉 所有工作流程步骤都已成功执行完成！\n"
        elif len(successful_steps) > len(failed_steps):
            report += (
                f"✅ 大部分步骤执行成功，建议检查失败的{len(failed_steps)}个步骤。\n"
            )
        else:
            report += f"⚠️ 多个步骤执行失败，建议检查配置和网络连接。\n"

        report += f"\n基于智能分类的动态工作流程为用户目标'{user_goal}'提供了个性化的执行方案。\n"

        return report.strip()

    def _register_tools(self):
        """注册主代理工具"""

        @self.agent.tool
        async def delegate_search_task(
            ctx: RunContext[MainAgentDependencies],
            query: str,
            search_type: Literal[
                "comprehensive", "jina_search", "arxiv"
            ] = "comprehensive",
            priority: TaskPriority = TaskPriority.MEDIUM,
        ) -> Dict[str, Any]:
            """委派搜索任务给搜索专家

            Args:
                query: 搜索查询
                search_type: 搜索类型
                priority: 任务优先级

            Returns:
                搜索任务结果
            """
            task_id = None
            try:
                # 创建搜索任务
                task_id = self._generate_task_id()
                task = TaskItem(
                    id=task_id,
                    title=f"搜索任务: {query[:50]}...",
                    description=f"执行{search_type}搜索: {query}",
                    priority=priority,
                    assigned_agent=AgentRole.SEARCH_SPECIALIST,
                    status="in_progress",
                )
                self.tasks[task_id] = task

                if config.project.debug_mode:
                    info(f"委派搜索任务 {task_id}", query=query)

                # 执行搜索
                search_deps = create_search_dependencies(
                    jina_api_key=config.embedding.api_key,
                    max_results=config.search.max_results,
                    enable_content_extraction=config.search.enable_content_extraction,
                )

                # 使用搜索代理执行搜索
                result = await ctx.deps.search_agent.search(
                    query, search_deps, search_type
                )

                # 更新任务状态
                task.status = "completed"
                task.results = {"search_result": result}
                task.updated_at = datetime.now()

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "search_type": search_type,
                    "agent": "search_specialist",
                }

            except Exception as e:
                # 更新任务状态为失败
                if task_id and task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].results = {"error": str(e)}
                    self.tasks[task_id].updated_at = datetime.now()

                if config.project.debug_mode:
                    error(f"搜索任务失败: {e}")

                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "agent": "search_specialist",
                }

        @self.agent.tool
        async def delegate_review_task(
            ctx: RunContext[MainAgentDependencies],
            group: str,
            bp_content: str,
            priority: TaskPriority = TaskPriority.HIGH,
        ) -> Dict[str, Any]:
            """
            委派商业计划书（BP）评审任务给评审专家。

            Args:
                group: 参赛赛道和组别
                bp_content: 商业计划书的完整内容
                priority: 任务优先级

            Returns:
                评审任务的结果
            """
            task_id = None
            try:
                # 验证赛道组别
                try:
                    competition_group = CompetitionGroup(group)
                except ValueError:
                    return {
                        "task_id": None,
                        "status": "failed",
                        "error": f"无效的赛道组别: {group}。有效选项为: {[g.value for g in CompetitionGroup]}",
                        "agent": "bp_reviewer",
                    }

                # 创建评审任务
                task_id = self._generate_task_id()
                task = TaskItem(
                    id=task_id,
                    title=f"BP评审任务: {group}",
                    description=f"对商业计划书进行评审",
                    priority=priority,
                    assigned_agent=AgentRole.MAIN_ASSISTANT,  # 假设由主助手协调
                    status="in_progress",
                )
                self.tasks[task_id] = task

                if config.project.debug_mode:
                    info(f"委派BP评审任务 {task_id}", group=group)

                # 执行评审
                result = await ctx.deps.bp_reviewer_agent.review(
                    competition_group, bp_content
                )

                # 更新任务状态
                task.status = "completed"
                task.results = {"review_result": result}
                task.updated_at = datetime.now()

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "group": group,
                    "agent": "bp_reviewer",
                }

            except Exception as e:
                if task_id and task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].results = {"error": str(e)}
                    self.tasks[task_id].updated_at = datetime.now()

                if config.project.debug_mode:
                    error(f"BP评审任务失败: {e}")

                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "agent": "bp_reviewer",
                }

        @self.agent.tool
        async def delegate_knowledge_task(
            ctx: RunContext[MainAgentDependencies],
            action: Literal["add", "search", "update", "analyze", "stats", "store"],
            content: str,
            knowledge_type: Optional[
                Literal["project_memory", "external_research"]
            ] = None,
            priority: TaskPriority = TaskPriority.MEDIUM,
        ) -> Dict[str, Any]:
            """委派知识管理任务给知识管理者

            Args:
                action: 执行的动作 (add, search, update, analyze, stats, store)
                content: 内容或查询
                knowledge_type: 知识类型
                priority: 任务优先级

            Returns:
                知识管理任务结果
            """
            task_id = None
            try:
                # 将store操作映射为add操作
                actual_action = "add" if action == "store" else action

                # 创建知识管理任务
                task_id = self._generate_task_id()
                task = TaskItem(
                    id=task_id,
                    title=f"知识管理: {action}",
                    description=f"执行知识库{action}操作: {content[:100]}...",
                    priority=priority,
                    assigned_agent=AgentRole.KNOWLEDGE_MANAGER,
                    status="in_progress",
                )
                self.tasks[task_id] = task

                if config.project.debug_mode:
                    info(f"委派知识管理任务 {task_id}", action=action)

                # 执行知识管理操作
                result = None
                if actual_action == "add":
                    # 使用知识代理添加知识
                    if knowledge_type is None:
                        knowledge_type = "external_research"

                    # 如果是store操作，表示要存储搜索结果
                    if action == "store":
                        add_prompt = f"请使用add_knowledge工具将搜索结果存储到知识库：\n标题：搜索结果 - {content[:50]}\n内容：{content}\n类型：{knowledge_type}\n来源：search_result"
                    else:
                        add_prompt = f"请使用add_knowledge工具添加知识：\n标题：用户输入\n内容：{content}\n类型：{knowledge_type}\n来源：user_input"
                    agent_result = await ctx.deps.knowledge_agent.agent.run(
                        add_prompt, deps=ctx.deps.knowledge_deps
                    )
                    result = agent_result.output

                elif action == "search":
                    # 使用知识代理搜索
                    search_prompt = f"请使用search_knowledge工具搜索: {content}"
                    if knowledge_type:
                        search_prompt += f"，知识类型：{knowledge_type}"

                    agent_result = await ctx.deps.knowledge_agent.agent.run(
                        search_prompt, deps=ctx.deps.knowledge_deps
                    )
                    result = agent_result.output

                elif action == "stats":
                    # 获取知识库统计
                    stats_result = await ctx.deps.knowledge_agent.agent.run(
                        "请使用get_knowledge_stats工具获取知识库统计信息",
                        deps=ctx.deps.knowledge_deps,
                    )
                    result = stats_result.output

                elif action == "analyze":
                    # 分析知识库内容
                    result = await ctx.deps.knowledge_agent.manage_project_memory(
                        "analyze", content, ctx.deps.knowledge_deps
                    )
                else:
                    result = f"已执行{action}操作，内容：{content}"

                # 更新任务状态
                task.status = "completed"
                task.results = {"knowledge_result": result}
                task.updated_at = datetime.now()

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "action": action,
                    "agent": "knowledge_manager",
                }

            except Exception as e:
                # 更新任务状态为失败
                if task_id and task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].results = {"error": str(e)}
                    self.tasks[task_id].updated_at = datetime.now()

                if config.project.debug_mode:
                    error(f"知识管理任务失败: {e}")

                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "agent": "knowledge_manager",
                }

        @self.agent.tool
        async def create_task_plan(
            ctx: RunContext[MainAgentDependencies],
            user_goal: str,
            project_stage: str = "planning",
        ) -> Dict[str, Any]:
            """创建任务计划，将用户目标分解为具体任务

            Args:
                user_goal: 用户目标描述
                project_stage: 项目阶段，默认为planning

            Returns:
                任务计划和任务列表
            """
            try:
                if config.project.debug_mode:
                    info("创建任务计划", goal=user_goal)

                # 使用意图分类Agent进行分类
                classification = await self.intent_classifier.classify_intent(
                    user_goal, ctx.deps.project_context
                )

                if config.project.debug_mode:
                    print(f"🤖 智能分类结果: {classification}")

                # 限制执行序列长度，避免过度复杂
                execution_sequence = classification.execution_sequence[
                    : ctx.deps.max_steps_per_workflow
                ]

                # 生成任务计划文本
                plan_content = f"""
                ## 任务计划：{user_goal[:100]}
                
                **项目阶段：** {project_stage}
                **项目描述：** {self.project_context.description[:200]}
                **智能分析结果：** {classification.reasoning[:200]}
                
                **执行策略：**
                1. 执行序列：{" → ".join(execution_sequence)}
                2. 最终目标：{classification.final_goal[:100]}
                3. 任务优先级：{classification.priority}
                4. 限制步骤数：{len(execution_sequence)}
                
                **关键要点：**
                - 基于AI智能分析的任务分类
                - 动态适应不同类型的用户需求
                - 限制复杂度避免递归地狱
                """

                # 根据智能分类结果执行相应的任务（限制复杂度）
                created_tasks = []
                priority_map = {
                    "low": TaskPriority.LOW,
                    "medium": TaskPriority.MEDIUM,
                    "high": TaskPriority.HIGH,
                    "urgent": TaskPriority.URGENT,
                }
                task_priority = priority_map.get(
                    classification.priority, TaskPriority.MEDIUM
                )

                # 限制任务执行数量，避免过度复杂
                max_auto_tasks = 2
                executed_count = 0

                # 按执行序列执行任务
                for i, task_type in enumerate(execution_sequence):
                    if executed_count >= max_auto_tasks:
                        break

                    if task_type == "search":
                        search_query = classification.search_query[:200]  # 限制查询长度
                        search_result = await delegate_search_task(
                            ctx, search_query, "comprehensive", task_priority
                        )
                        created_tasks.append(search_result)
                        executed_count += 1

                    elif task_type == "knowledge_add":
                        # 简化知识添加逻辑
                        if "search" in execution_sequence[:i] and executed_count > 0:
                            # 如果前面有搜索，则存储搜索结果
                            knowledge_result = await delegate_knowledge_task(
                                ctx,
                                "store",
                                f"存储搜索结果：{classification.search_query[:100]}",
                                classification.knowledge_type,
                                TaskPriority.MEDIUM,
                            )
                        else:
                            knowledge_result = await delegate_knowledge_task(
                                ctx,
                                "add",
                                user_goal[:300],  # 限制内容长度
                                classification.knowledge_type,
                                task_priority,
                            )
                        created_tasks.append(knowledge_result)
                        executed_count += 1

                    elif task_type == "knowledge_search":
                        search_query = classification.search_query[:200]
                        knowledge_result = await delegate_knowledge_task(
                            ctx,
                            "search",
                            search_query,
                            classification.knowledge_type,
                            task_priority,
                        )
                        created_tasks.append(knowledge_result)
                        executed_count += 1

                    # 其他任务类型简化处理
                    elif (
                        task_type in ["analysis", "planning", "general"]
                        and executed_count == 0
                    ):
                        # 只在没有执行其他任务时执行这些任务
                        simple_result = await delegate_knowledge_task(
                            ctx,
                            "add",
                            f"{task_type}: {user_goal[:200]}",
                            "project_memory",
                            task_priority,
                        )
                        created_tasks.append(simple_result)
                        executed_count += 1

                return {
                    "plan": plan_content,
                    "executed_tasks": created_tasks,
                    "total_tasks": len(created_tasks),
                    "limited_execution": executed_count >= max_auto_tasks,
                    "classification": classification.model_dump(),
                    "status": "created_and_executed",
                }

            except Exception as e:
                error_msg = str(e)
                if config.project.debug_mode:
                    print(f"❌ 任务计划创建失败: {error_msg}")

                return {
                    "error": error_msg,
                    "status": "failed",
                    "plan": f"任务计划创建失败: {error_msg}",
                }

        @self.agent.tool
        async def execute_dynamic_workflow(
            ctx: RunContext[MainAgentDependencies],
            user_goal: str,
            workflow_context: Optional[Dict[str, Any]] = None,
            # 新增参数，用于接收预先计算的分类结果
            classification_input: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """动态执行工作流程 - 基于智能分类结果

            Args:
                user_goal: 用户目标描述
                workflow_context: 工作流程上下文信息
                classification_input: (可选) 预先计算的分类结果字典

            Returns:
                动态工作流程执行结果
            """
            try:
                # 递归深度检查
                if ctx.deps.current_recursion_depth >= ctx.deps.max_recursion_depth:
                    return {
                        "workflow_type": "dynamic",
                        "status": "failed",
                        "error": f"达到最大递归深度限制({ctx.deps.max_recursion_depth})，停止执行以防止无限循环",
                        "user_goal": user_goal,
                        "recursion_depth": ctx.deps.current_recursion_depth,
                    }

                # 检查任务循环
                task_signature = f"execute_dynamic_workflow:{hash(user_goal) % 10000}"
                if task_signature in ctx.deps.task_execution_stack:
                    return {
                        "workflow_type": "dynamic",
                        "status": "failed",
                        "error": f"检测到任务循环，停止执行。任务签名: {task_signature}",
                        "user_goal": user_goal,
                        "task_stack": ctx.deps.task_execution_stack,
                    }

                # 将当前任务加入执行栈
                ctx.deps.task_execution_stack.append(task_signature)
                ctx.deps.current_recursion_depth += 1

                if config.project.debug_mode:
                    print(
                        f"🔄 动态工作流程执行: {user_goal} [递归深度: {ctx.deps.current_recursion_depth}]"
                    )

                # 1. 智能分类用户意图 (如果未提供)
                if classification_input:
                    # 如果传入了分类结果，直接使用
                    classification = UserIntentClassification(**classification_input)
                    if config.project.debug_mode:
                        print(f"🤖 使用预计算的分类结果: {classification}")
                else:
                    # 否则，执行新的分类
                    if config.project.debug_mode:
                        print(f"🤖 未提供分类，执行新的意图分类...")
                    classification = await self.intent_classifier.classify_intent(
                        user_goal, ctx.deps.project_context
                    )

                if config.project.debug_mode:
                    print(f"🤖 智能分类结果: {classification}")

                # 2. 构建动态工作流程计划
                workflow_plan = await self._build_dynamic_workflow_plan(
                    classification, user_goal, ctx
                )

                # 3. 执行动态工作流程
                execution_results = await self._execute_workflow_steps(
                    workflow_plan["steps"], ctx
                )

                # 4. 生成工作流程报告
                workflow_report = await self._generate_workflow_report(
                    classification, workflow_plan, execution_results, user_goal
                )

                return {
                    "workflow_type": "dynamic",
                    "classification": classification.model_dump(),
                    "plan": workflow_plan,
                    "execution_results": execution_results,
                    "report": workflow_report,
                    "status": "completed",
                    "total_steps": len(workflow_plan["steps"]),
                    "successful_steps": len(
                        [r for r in execution_results if r.get("status") == "completed"]
                    ),
                    "recursion_depth": ctx.deps.current_recursion_depth,
                }

            except Exception as e:
                error_msg = str(e)
                if config.project.debug_mode:
                    print(f"❌ 动态工作流程执行失败: {error_msg}")

                return {
                    "workflow_type": "dynamic",
                    "status": "failed",
                    "error": error_msg,
                    "user_goal": user_goal,
                    "recursion_depth": ctx.deps.current_recursion_depth,
                }
            finally:
                # 清理递归状态
                if (
                    ctx.deps.task_execution_stack
                    and task_signature in ctx.deps.task_execution_stack
                ):
                    ctx.deps.task_execution_stack.remove(task_signature)
                ctx.deps.current_recursion_depth = max(
                    0, ctx.deps.current_recursion_depth - 1
                )

                if config.project.debug_mode:
                    print(
                        f"🧹 清理递归状态，当前深度: {ctx.deps.current_recursion_depth}"
                    )

        @self.agent.tool
        async def get_system_status(
            ctx: RunContext[MainAgentDependencies],
        ) -> SystemSummary:
            """获取系统状态摘要

            Returns:
                系统状态信息
            """
            try:
                # 统计任务状态
                total_tasks = len(self.tasks)
                completed_tasks = len(
                    [t for t in self.tasks.values() if t.status == "completed"]
                )
                active_tasks = len(
                    [
                        t
                        for t in self.tasks.values()
                        if t.status in ["pending", "in_progress"]
                    ]
                )

                # 获取最近活动
                recent_tasks = sorted(
                    self.tasks.values(), key=lambda x: x.updated_at, reverse=True
                )[:5]

                recent_activities = [
                    f"{task.title} - {task.status}" for task in recent_tasks
                ]

                # 生成建议
                recommendations = []
                if active_tasks == 0:
                    recommendations.append("可以开始新的研究任务")
                if completed_tasks > 0:
                    recommendations.append("建议整理已完成的任务成果")

                # 尝试获取知识库统计
                knowledge_entries = 0
                try:
                    # 使用委派任务的方式获取知识库统计
                    stats_task_result = await delegate_knowledge_task(
                        ctx, "stats", "获取知识库统计信息", None, TaskPriority.LOW
                    )

                    if stats_task_result.get("status") == "completed":
                        # 尝试从结果中提取统计信息
                        result_str = str(stats_task_result.get("result", ""))
                        if "total_entries" in result_str.lower():
                            # 简单估算，实际项目中应该解析JSON
                            knowledge_entries = 10

                except Exception as stats_error:
                    if config.project.debug_mode:
                        print(f"获取知识库统计失败: {stats_error}")

                return SystemSummary(
                    active_tasks=active_tasks,
                    completed_tasks=completed_tasks,
                    knowledge_entries=knowledge_entries,
                    recent_activities=recent_activities,
                    recommendations=recommendations,
                )

            except Exception as e:
                if config.project.debug_mode:
                    print(f"获取系统状态失败: {e}")

                return SystemSummary(
                    active_tasks=0,
                    completed_tasks=0,
                    knowledge_entries=0,
                    recent_activities=[f"获取状态失败: {e}"],
                    recommendations=["请检查系统配置"],
                )

        @self.agent.tool
        async def execute_scheduled_task(
            ctx: RunContext[MainAgentDependencies],
            task_type: Literal[
                "daily_summary", "project_review", "knowledge_update"
            ] = "daily_summary",
        ) -> Dict[str, Any]:
            """执行定期任务

            Args:
                task_type: 任务类型

            Returns:
                执行结果
            """
            try:
                if task_type == "daily_summary":
                    # 生成每日摘要
                    summary = await get_system_status(ctx)

                    summary_content = f"""
                    ## CreatPartner 每日项目摘要
                    
                    ### 项目信息
                    - 项目名称: {self.project_context.name}
                    - 当前阶段: {self.project_context.stage.value}
                    
                    ### 任务统计
                    - 活跃任务: {summary.active_tasks}
                    - 已完成任务: {summary.completed_tasks}
                    - 知识库条目: {summary.knowledge_entries}
                    
                    ### 近期活动
                    {chr(10).join(f"- {activity}" for activity in summary.recent_activities)}
                    
                    ### 建议
                    {chr(10).join(f"- {rec}" for rec in summary.recommendations)}
                    """

                    # 将摘要保存到知识库
                    await delegate_knowledge_task(
                        ctx, "add", summary_content, "project_memory", TaskPriority.LOW
                    )

                    return {
                        "task_type": task_type,
                        "status": "completed",
                        "summary": summary_content,
                        "timestamp": datetime.now().isoformat(),
                    }

                # 其他定期任务类型的处理...
                return {
                    "task_type": task_type,
                    "status": "not_implemented",
                    "message": f"任务类型 {task_type} 尚未实现",
                }

            except Exception as e:
                return {"task_type": task_type, "status": "failed", "error": str(e)}

    async def chat(self, message: str, session_id: str = "default") -> AgentResponse:
        """与主代理进行对话"""
        try:
            # 创建依赖配置
            knowledge_deps = KnowledgeDependencies(
                mongodb_uri=config.database.mongodb_uri,
                database_name=config.database.database_name,
                jina_api_key=config.embedding.api_key,
            )

            deps = MainAgentDependencies(
                project_context=self.project_context,
                search_agent=self.search_agent,
                knowledge_agent=self.knowledge_agent,
                bp_reviewer_agent=self.bp_reviewer_agent,  # 新增
                knowledge_deps=knowledge_deps,
                session_id=session_id,
                current_recursion_depth=0,  # 初始化递归深度
                max_recursion_depth=3,  # 设置最大递归深度
                task_execution_stack=[],  # 初始化任务执行栈
            )

            # 使用意图分类Agent进行分类
            classification = await self.intent_classifier.classify_intent(
                message, self.project_context
            )

            if config.project.debug_mode:
                print(f"🤖 对话智能分类: {classification}")

            # 根据分类结果决定处理方式
            execution_sequence = classification.execution_sequence

            # 检测复杂任务（需要多步骤执行），使用动态工作流程
            is_complex_task = (
                len(execution_sequence) > 1
                or execution_sequence[0]
                in ["search", "analysis", "planning", "review_bp"]
                or (
                    "搜索" in message
                    and ("存储" in message or "保存" in message or "知识库" in message)
                )
                or ("评审" in message and "商业计划书" in message)
            )

            if is_complex_task:
                # 使用动态工作流程处理复杂任务
                # 将分类结果传递给工作流，避免重复分类
                classification_json = json.dumps(
                    classification.model_dump(), ensure_ascii=False
                )
                prompt = f"""
                用户目标是: '{message}'
                我已经对这个目标进行了初步的智能分类，结果如下:
                {classification_json}

                请使用 `execute_dynamic_workflow` 工具来执行这个任务。
                将用户目标 '{message}' 作为 `user_goal` 参数。
                将上面的分类结果作为 `classification_input` 参数传递给工具。
                """
                result = await self.agent.run(prompt, deps=deps)

                return AgentResponse(
                    agent_role=AgentRole.MAIN_ASSISTANT,
                    content=result.output,
                    confidence=0.95,
                    additional_data={
                        "classification": classification.model_dump(),
                        "workflow_type": "dynamic",
                    },
                )

            elif execution_sequence[0] == "general":
                # 对于一般性对话，直接处理
                result = await self.agent.run(message, deps=deps)

                return AgentResponse(
                    agent_role=AgentRole.MAIN_ASSISTANT,
                    content=result.output,
                    confidence=0.8,
                    additional_data={"classification": classification.model_dump()},
                )

            else:
                # 使用单一任务工具处理简单任务
                primary_task = execution_sequence[0]
                if primary_task == "search":
                    result = await self.agent.run(
                        f"请使用delegate_search_task工具搜索：{classification.search_query}",
                        deps=deps,
                    )
                elif primary_task == "knowledge_add":
                    result = await self.agent.run(
                        f"请使用delegate_knowledge_task工具添加知识：{message}",
                        deps=deps,
                    )
                elif primary_task == "review_bp":
                    # 这是一个简化的调用，实际应用中BP内容和赛道需要从用户输入中提取
                    result = await self.agent.run(
                        f"请使用delegate_review_task工具评审商业计划书。赛道是'高教主赛道-创意组'，内容是：'{message}'",
                        deps=deps,
                    )
                elif primary_task == "knowledge_search":
                    result = await self.agent.run(
                        f"请使用delegate_knowledge_task工具搜索知识：{classification.search_query}",
                        deps=deps,
                    )
                else:
                    # 默认使用动态工作流程
                    result = await self.agent.run(
                        f"请使用execute_dynamic_workflow工具处理用户目标：{message}",
                        deps=deps,
                    )

                return AgentResponse(
                    agent_role=AgentRole.MAIN_ASSISTANT,
                    content=result.output,
                    confidence=0.9,
                    additional_data={"classification": classification.model_dump()},
                )

        except Exception as e:
            error_msg = f"处理请求时发生错误: {str(e)}"
            if config.project.debug_mode:
                print(f"❌ 对话处理失败: {e}")

            return AgentResponse(
                agent_role=AgentRole.MAIN_ASSISTANT, content=error_msg, confidence=0.1
            )

    def chat_sync(self, message: str, session_id: str = "default") -> AgentResponse:
        """同步版本的对话接口"""
        return asyncio.run(self.chat(message, session_id))

    def get_project_summary(self) -> Dict[str, Any]:
        """获取项目摘要"""
        return {
            "project": self.project_context.dict(),
            "tasks": {
                "total": len(self.tasks),
                "completed": len(
                    [t for t in self.tasks.values() if t.status == "completed"]
                ),
                "active": len(
                    [
                        t
                        for t in self.tasks.values()
                        if t.status in ["pending", "in_progress"]
                    ]
                ),
                "failed": len([t for t in self.tasks.values() if t.status == "failed"]),
            },
            "agents": {
                "search_agent": "已初始化",
                "knowledge_agent": "已初始化",
                "bp_reviewer_agent": "已初始化",  # 新增
                "main_agent": "已初始化",
            },
        }


# 工厂函数
def create_main_agent(
    model_name: str = None,
    project_name: str = "创新创业项目",
    project_description: str = "基于AI技术的创新创业项目",
    project_stage: ProjectStage = ProjectStage.PLANNING,
) -> MainAgent:
    """创建主代理实例"""
    project_context = ProjectContext(
        name=project_name,
        description=project_description,
        stage=project_stage,
        keywords=["AI", "创新", "创业"],
    )

    return MainAgent(model_name, project_context)


# 示例用法
async def main():
    """主代理演示"""
    print("🚀 CreatPartner 主代理演示")
    print("=" * 50)

    # 创建主代理
    agent = create_main_agent(
        project_name="AI教育助手",
        project_description="基于大语言模型的个性化教育助手",
        project_stage=ProjectStage.RESEARCH,
    )

    # 演示对话
    test_queries = [
        "你好，请介绍一下CreatPartner系统",
        "我需要调研AI在教育领域的应用现状",
        "请帮我制定一个AI教育助手项目的研究计划",
        "请搜索相关的学术论文和市场报告",
        "请总结目前的项目进展",
    ]

    for query in test_queries:
        print(f"\n用户: {query}")
        try:
            response = await agent.chat(query)
            print(f"助手: {response.content[:300]}...")
            print(f"置信度: {response.confidence}")
        except Exception as e:
            print(f"❌ 处理失败: {e}")

    # 显示项目摘要
    summary = agent.get_project_summary()
    print(f"\n📊 项目摘要:")
    print(f"项目名称: {summary['project']['name']}")
    print(f"项目阶段: {summary['project']['stage']}")
    print(f"总任务数: {summary['tasks']['total']}")
    print(f"已完成: {summary['tasks']['completed']}")
    print(f"进行中: {summary['tasks']['active']}")


if __name__ == "__main__":
    asyncio.run(main())
