import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import streamlit as st
import uuid
import pandas as pd
import logfire
from contextlib import contextmanager

st.set_page_config(
    page_title="CreatPartner - AI创新创业助手",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/nkanf-dev/CreatPartner",
        "Report a bug": "https://github.com/nkanf-dev/CreatPartner/issues",
        "About": """
        # CreatPartner 🚀
        
        AI驱动的创新创业项目助手
        
        **功能特点:**
        - 🤖 智能对话助手
        - 📚 知识库管理
        - 🔍 资料搜索
        - 📋 项目管理
        
        **版本:** v1.1.0
        **作者:** CreatPartner Team
        
        **技术栈:**
        - Agent框架: pydantic-ai
        - 长期记忆: MongoDB + Jina Embedding
        - 前端: Streamlit
        - 文档解析: MinerU
        """,
    },
)

# 导入核心组件
from config import config, validate_config
from main_agent import (
    MainAgent,
    create_main_agent,
    ProjectContext,
    ProjectStage,
    AgentRole,
    AgentResponse,
)

# 导入日志系统
from logger import (
    get_logger,
    info,
    error,
    success,
    warning,
    debug,
    set_broadcast_function,
)

# 初始化日志
logger = get_logger()

# 日志广播系统
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []


def broadcast_to_ui(level: str, message: str, **kwargs):
    """将日志消息广播到UI"""
    if len(st.session_state.log_messages) > 100:  # 限制日志条数
        st.session_state.log_messages = st.session_state.log_messages[-50:]

    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "kwargs": kwargs,
    }
    st.session_state.log_messages.append(log_entry)


# 设置日志广播函数
set_broadcast_function(broadcast_to_ui)

info("CreatPartner应用启动")


class RealTimeStatusManager:
    """实时状态管理器，用于在UI中显示Agent执行状态"""

    def __init__(self):
        logfire.configure()
        logfire.instrument_pydantic_ai()
        self.current_status = ""
        self.current_progress = 0
        self.status_history = []
        self.is_active = False

    def update_status(self, status: str, progress: int = None):
        """更新当前状态"""
        self.current_status = status
        if progress is not None:
            self.current_progress = progress

        self.status_history.append(
            {
                "status": status,
                "progress": self.current_progress,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 限制历史记录长度
        if len(self.status_history) > 50:
            self.status_history = self.status_history[-50:]

    def start_task(self, task_name: str):
        """开始任务"""
        self.is_active = True
        self.update_status(f"🚀 开始任务: {task_name}", 0)

    def finish_task(self, success: bool = True):
        """结束任务"""
        self.is_active = False
        status = "✅ 任务完成!" if success else "❌ 任务失败!"
        self.update_status(status, 100)

    def clear(self):
        """清除状态"""
        self.current_status = ""
        self.current_progress = 0
        self.status_history = []
        self.is_active = False


class TaskStatusCapture:
    """任务状态捕获器 - 捕获主代理执行过程中的状态信息"""

    def __init__(self):
        self.status_queue = []
        self.current_task = None
        self.is_capturing = False

    def start_capture(self, task_name: str):
        """开始捕获状态"""
        self.is_capturing = True
        self.current_task = task_name
        self.status_queue = []
        self.add_status(f"🚀 开始执行: {task_name}")

    def add_status(self, status: str, emoji: str = None):
        """添加状态信息"""
        if self.is_capturing:
            timestamp = datetime.now().strftime("%H:%M:%S")
            status_item = {
                "timestamp": timestamp,
                "status": status,
                "emoji": emoji or self._extract_emoji(status),
            }
            self.status_queue.append(status_item)

    def _extract_emoji(self, status: str):
        """从状态字符串中提取emoji"""
        for char in status:
            if ord(char) > 127:  # 简单检测非ASCII字符（包括emoji）
                return char
        return "⚙️"

    def stop_capture(self):
        """停止捕获"""
        self.is_capturing = False
        if self.current_task:
            self.add_status(f"✅ 完成任务: {self.current_task}")
        self.current_task = None

    def get_latest_status(self):
        """获取最新状态"""
        return self.status_queue[-1] if self.status_queue else None

    def get_all_statuses(self):
        """获取所有状态"""
        return self.status_queue.copy()

    def clear(self):
        """清除状态"""
        self.status_queue = []
        self.current_task = None
        self.is_capturing = False


class StreamlitStatusUpdater:
    """Streamlit状态更新器 - 在UI中实时显示最新日志状态"""

    def __init__(self, status_container, progress_container=None):
        self.status_container = status_container
        self.progress_container = progress_container
        self.current_progress = 0
        self.max_progress = 100

    def update_with_latest_logs(self, max_logs: int = 5):
        """使用最新的日志消息更新UI状态"""
        if not self.status_container:
            return

        # 获取最新的日志消息
        recent_logs = (
            st.session_state.log_messages[-max_logs:]
            if st.session_state.log_messages
            else []
        )

        if recent_logs:
            # 构建显示文本
            log_text = ""
            for log_entry in recent_logs:
                timestamp = log_entry["timestamp"]
                level = log_entry["level"]
                message = log_entry["message"]

                # 根据日志级别选择显示样式
                if level == "ERROR":
                    emoji = "❌"
                elif level == "SUCCESS":
                    emoji = "✅"
                elif level == "WARNING":
                    emoji = "⚠️"
                elif level == "DEBUG":
                    emoji = "🔧"
                else:
                    emoji = "ℹ️"

                log_text += f"`[{timestamp}]` {emoji} {message}\n\n"

            # 显示最新日志
            self.status_container.info(log_text)
        else:
            self.status_container.info("🤖 AI助手准备就绪...")

    def update_status(self, status: str, progress: int = None):
        """更新UI状态（兼容旧接口）"""
        # 记录状态到日志系统
        info(status)

        # 更新显示
        self.update_with_latest_logs()

        # 更新进度条
        if progress is not None and self.progress_container:
            self.current_progress = min(progress, self.max_progress)
            self.progress_container.progress(self.current_progress)

    def show_success(self, message: str):
        """显示成功状态"""
        success(message)
        self.update_with_latest_logs()
        if self.progress_container:
            self.progress_container.progress(100)

    def show_error(self, message: str):
        """显示错误状态"""
        error(message)
        self.update_with_latest_logs()


class EnhancedMainAgentWithMonitoring:
    """带监控的增强主代理 - 可以捕获并显示执行过程状态"""

    def __init__(self, main_agent: MainAgent):
        self.main_agent = main_agent
        self.status_capture = TaskStatusCapture()
        self.ui_updater = None

    def set_ui_updater(self, ui_updater: StreamlitStatusUpdater):
        """设置UI更新器"""
        self.ui_updater = ui_updater

    async def chat_with_monitoring(self, message: str, session_id: str = "default"):
        """带监控的对话方法"""
        try:
            # 开始捕获状态
            self.status_capture.start_capture("处理用户请求")

            # 使用日志系统记录处理步骤
            info("🤖 AI助手正在分析您的请求...")
            if self.ui_updater:
                self.ui_updater.update_with_latest_logs()

            # 使用意图分类获取执行计划
            info("🧠 正在进行意图分析...")
            classification_result = (
                await self.main_agent.intent_classifier.classify_intent(
                    message, self.main_agent.project_context
                )
            )

            execution_sequence = classification_result.execution_sequence
            info(f"📋 制定执行计划: {' → '.join(execution_sequence)}")
            if self.ui_updater:
                self.ui_updater.update_with_latest_logs()

            # 记录步骤执行状态
            if classification_result:
                self._log_execution_status(classification_result)

            # 执行实际对话
            info("🔄 正在处理用户请求...")
            if self.ui_updater:
                self.ui_updater.update_with_latest_logs()

            response = await self.main_agent.chat(message, session_id)

            # 完成
            success("✅ 用户请求处理完成!")
            if self.ui_updater:
                self.ui_updater.show_success("✅ 处理完成!")

            self.status_capture.stop_capture()
            return response

        except Exception as e:
            error_msg = f"❌ 处理用户请求失败: {str(e)}"
            error(error_msg)
            if self.ui_updater:
                self.ui_updater.show_error(error_msg)
            self.status_capture.stop_capture()
            raise e

    def _log_execution_status(self, classification_result):
        """记录执行状态到日志系统"""
        execution_sequence = classification_result.execution_sequence

        # 状态映射
        status_map = {
            "search": "🔍 搜索外部资料",
            "knowledge_add": "📚 添加到知识库",
            "knowledge_search": "🔎 搜索知识库",
            "analysis": "📊 分析处理",
            "planning": "📋 制定计划",
            "general": "💬 常规对话",
        }

        # 记录执行步骤到日志
        for step_type in execution_sequence:
            step_name = status_map.get(step_type, f"⚙️ {step_type}")
            info(f"准备执行: {step_name}")
            # 同时记录到状态捕获器
            self.status_capture.add_status(step_name)

        # 触发UI更新
        if self.ui_updater:
            self.ui_updater.update_with_latest_logs()

    def get_status_history(self):
        """获取状态历史"""
        return self.status_capture.get_all_statuses()

    def __getattr__(self, name):
        """代理其他方法到原始main_agent"""
        return getattr(self.main_agent, name)


# 全局状态管理器
if "status_manager" not in st.session_state:
    st.session_state.status_manager = RealTimeStatusManager()


@contextmanager
def status_context(task_name: str):
    """状态上下文管理器"""
    try:
        st.session_state.status_manager.start_task(task_name)
        yield st.session_state.status_manager
    except Exception as e:
        st.session_state.status_manager.update_status(f"❌ 错误: {str(e)}", None)
        st.session_state.status_manager.finish_task(False)
        raise
    finally:
        if st.session_state.status_manager.is_active:
            st.session_state.status_manager.finish_task(True)


def load_user_state_from_storage():
    """从浏览器本地存储加载用户状态"""
    try:
        # 这里我们使用session state来模拟本地存储
        # 在实际应用中，可以通过streamlit-js-eval等库与浏览器交互
        pass
    except Exception as e:
        st.error(f"加载用户状态失败: {e}")


def save_user_state_to_storage():
    """保存用户状态到浏览器本地存储"""
    try:
        # 构建要保存的状态数据
        state_data = {
            "project": st.session_state.project,
            "chat_history": st.session_state.chat_history[-50:],  # 只保存最近50条
            "app_settings": st.session_state.app_settings,
            "last_save": datetime.now().isoformat(),
        }

        # 模拟保存到 st.session_state
        st.session_state.user_state_storage = state_data

    except Exception as e:
        st.error(f"保存用户状态失败: {e}")


def reset_project():
    """重置当前项目为新项目"""
    st.session_state.project = {
        "id": str(uuid.uuid4()),
        "name": "我的创新项目",
        "description": "一个全新的创新创业项目。",
        "stage": ProjectStage.PLANNING.value,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "tags": [],
        "team_members": [],
        "progress": 0,
        "status": "active",
    }

    # 更新项目上下文并保存
    update_project_context()
    save_user_state_to_storage()

    # 重置agent和聊天记录
    st.session_state.main_agent = None
    st.session_state.chat_history = []


def update_project_context():
    """从当前项目更新项目上下文"""
    if "project" in st.session_state:
        current_project = st.session_state.project

        st.session_state.project_context.name = current_project.get(
            "name", "未命名项目"
        )
        st.session_state.project_context.description = current_project.get(
            "description", ""
        )
        st.session_state.project_context.stage = ProjectStage(
            current_project.get("stage", ProjectStage.PLANNING.value)
        )

        if current_project.get("created_at"):
            st.session_state.project_context.created_at = datetime.fromisoformat(
                current_project["created_at"]
            )
        if current_project.get("updated_at"):
            st.session_state.project_context.updated_at = datetime.fromisoformat(
                current_project["updated_at"]
            )


def update_project_from_context():
    """从项目上下文更新当前项目"""
    if "project" in st.session_state:
        current_project = st.session_state.project

        current_project["name"] = st.session_state.project_context.name
        current_project["description"] = st.session_state.project_context.description
        current_project["stage"] = st.session_state.project_context.stage.value
        current_project["updated_at"] = datetime.now().isoformat()

        # 保存状态
        save_user_state_to_storage()


def init_session_state():
    """初始化session state"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if "project" not in st.session_state:
        st.session_state.project = {
            "id": str(uuid.uuid4()),
            "name": "我的创新项目",
            "description": "一个全新的创新创业项目。",
            "stage": ProjectStage.PLANNING.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": [],
            "team_members": [],
            "progress": 0,
            "status": "active",
        }

    if "main_agent" not in st.session_state:
        st.session_state.main_agent = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "project_context" not in st.session_state:
        st.session_state.project_context = ProjectContext()

    if "app_settings" not in st.session_state:
        st.session_state.app_settings = {
            "auto_save_interval": 30,
            "max_chat_history": 100,
            "enable_analytics": True,
            "debug_mode": False,
        }

    load_user_state_from_storage()

    if "project" not in st.session_state or not st.session_state.project:
        reset_project()


def load_agent():
    """加载或创建主代理"""
    update_project_context()

    if st.session_state.main_agent is None:
        info("开始初始化AI助手系统")
        with st.spinner("正在初始化AI助手系统..."):
            try:
                base_agent = create_main_agent(
                    project_name=st.session_state.project_context.name,
                    project_description=st.session_state.project_context.description,
                    project_stage=st.session_state.project_context.stage,
                )
                st.session_state.main_agent = EnhancedMainAgentWithMonitoring(
                    base_agent
                )
                success(
                    "AI助手系统初始化成功",
                    project=st.session_state.project_context.name,
                )
                st.success("✅ AI助手系统初始化成功！")
                return True
            except Exception as e:
                error("AI助手系统初始化失败", error=str(e))
                st.error(f"❌ 初始化失败: {e}")
                return False
    return True


def render_header():
    """渲染页面头部"""
    # 自定义CSS样式
    st.markdown(
        """
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: bold;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp > div:first-child {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #a8edea 0%, #fed6e3 100%);
    }
    /* 美化聊天界面 */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    /* 状态显示样式 */
    .status-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="main-header">
        <h1>🚀 CreatPartner</h1>
        <p>AI驱动的创新创业项目助手</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("📁 项目管理")

        project_name = st.session_state.project.get("name", "未命名项目")
        st.info(f"当前项目: **{project_name}**")

        if st.button("🔄 重置项目"):
            reset_project()
            st.rerun()

        st.divider()
        st.header("⚙️ 应用设置")

        if st.button("🔍 检查配置"):
            with st.spinner("检查配置中..."):
                if validate_config():
                    st.success("✅ 配置检查通过")
                else:
                    st.error("❌ 配置检查失败")


def render_project_management():
    """渲染项目管理界面"""
    st.subheader("🎯 项目配置")

    project_name = st.text_input(
        "项目名称",
        value=st.session_state.project_context.name,
        key="project_name_input",
    )

    project_description = st.text_area(
        "项目描述",
        value=st.session_state.project_context.description,
        key="project_desc_input",
        height=100,
    )

    project_stage = st.selectbox(
        "项目阶段",
        options=[stage.value for stage in ProjectStage],
        index=[stage.value for stage in ProjectStage].index(
            st.session_state.project_context.stage.value
        ),
        format_func=lambda x: {
            "planning": "📝 规划阶段",
            "research": "🔍 调研阶段",
            "development": "⚙️ 开发阶段",
            "testing": "🧪 测试阶段",
            "deployment": "🚀 部署阶段",
            "competition": "🏆 比赛阶段",
        }.get(x, x),
        key="project_stage_input",
    )

    if st.button("💾 保存项目配置"):
        st.session_state.project_context.name = project_name
        st.session_state.project_context.description = project_description
        st.session_state.project_context.stage = ProjectStage(project_stage)
        st.session_state.project_context.updated_at = datetime.now()
        update_project_from_context()
        st.session_state.main_agent = None  # 重置代理以应用新配置
        st.success("项目配置已保存！")
        st.rerun()


def render_task_management():
    """渲染任务管理界面"""
    st.header("📋 任务管理")

    if not st.session_state.main_agent:
        st.warning("请先初始化AI助手系统")
        return

    # 获取任务列表
    tasks = st.session_state.main_agent.tasks

    if not tasks:
        st.info("暂无任务。您可以通过对话创建新任务。")
        return

    # 任务统计
    col1, col2, col3, col4 = st.columns(4)

    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks.values() if t.status == "completed"])
    active_tasks = len(
        [t for t in tasks.values() if t.status in ["pending", "in_progress"]]
    )
    failed_tasks = len([t for t in tasks.values() if t.status == "failed"])

    with col1:
        st.metric("总任务", total_tasks)

    with col2:
        st.metric("已完成", completed_tasks)

    with col3:
        st.metric("进行中", active_tasks)

    with col4:
        st.metric("失败", failed_tasks)

    st.divider()

    # 任务列表
    st.subheader("任务列表")

    # 状态过滤
    status_filter = st.selectbox(
        "筛选状态",
        options=["全部", "pending", "in_progress", "completed", "failed"],
        key="task_mgmt_status_filter",
    )

    # 优先级过滤
    priority_filter = st.selectbox(
        "筛选优先级",
        options=["全部", "low", "medium", "high", "urgent"],
        key="task_mgmt_priority_filter",
    )

    # 过滤任务
    filtered_tasks = list(tasks.values())

    if status_filter != "全部":
        filtered_tasks = [t for t in filtered_tasks if t.status == status_filter]

    if priority_filter != "全部":
        filtered_tasks = [
            t for t in filtered_tasks if t.priority.value == priority_filter
        ]

    # 按更新时间排序
    filtered_tasks.sort(key=lambda x: x.updated_at, reverse=True)

    # 显示任务
    for task in filtered_tasks:
        status_class = f"status-{task.status.replace('_', '-')}"

        st.markdown(
            f"""
        <div class="task-item {status_class}">
            <h4>{task.title}</h4>
            <p><strong>状态:</strong> {task.status} | 
               <strong>优先级:</strong> {task.priority.value} | 
               <strong>负责Agent:</strong> {task.assigned_agent.value}</p>
            <p><strong>描述:</strong> {task.description}</p>
            <p><small>创建时间: {task.created_at.strftime("%Y-%m-%d %H:%M:%S")}</small></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # 显示任务结果
        if task.results:
            with st.expander(f"查看任务结果 - {task.title}"):
                st.json(task.results)


def render_knowledge_management():
    """渲染知识库管理界面"""
    st.header("📚 知识库管理")

    if not st.session_state.main_agent:
        st.warning("请先初始化AI助手系统")
        return

    # 知识库操作选项卡
    tab1, tab2, tab3 = st.tabs(["📝 添加知识", "🔍 搜索知识", "📊 知识统计"])

    with tab1:
        st.subheader("添加知识到知识库")

        knowledge_type = st.selectbox(
            "知识类型",
            options=["project_memory", "external_research"],
            format_func=lambda x: "项目记忆" if x == "project_memory" else "外部资料",
        )

        title = st.text_input("知识标题")
        content = st.text_area("知识内容", height=200)
        source = st.text_input("来源", value="user_input")
        tags = st.text_input("标签 (用逗号分隔)")

        if st.button("💾 添加知识"):
            if title and content:
                with st.spinner("正在添加知识..."):
                    try:
                        # 直接调用 knowledge_agent 添加知识
                        async def add_knowledge_async():
                            tag_list = [
                                tag.strip() for tag in tags.split(",") if tag.strip()
                            ]
                            await st.session_state.main_agent.knowledge_agent.add_document(
                                title=title,
                                content=content,
                                source=source,
                                tags=tag_list,
                                knowledge_base=knowledge_type,
                            )

                        asyncio.run(add_knowledge_async())
                        st.success("✅ 知识添加成功")

                    except Exception as e:
                        st.error(f"❌ 添加失败: {e}")
            else:
                st.warning("请填写标题和内容")

    with tab2:
        st.subheader("搜索知识库")

        search_query = st.text_input("搜索查询")
        search_type = st.selectbox(
            "搜索范围",
            options=["全部", "project_memory", "external_research"],
            format_func=lambda x: "全部知识库"
            if x == "全部"
            else ("项目记忆" if x == "project_memory" else "外部资料"),
        )

        if st.button("🔍 搜索") and search_query:
            with st.spinner("正在搜索知识库..."):
                try:
                    search_message = f"请在知识库中搜索：{search_query}"
                    if search_type != "全部":
                        search_message += f"，搜索范围：{search_type}"

                    response = asyncio.run(
                        st.session_state.main_agent.chat(search_message)
                    )
                    st.success("✅ 搜索完成")
                    st.write(response.content)

                except Exception as e:
                    st.error(f"❌ 搜索失败: {e}")

    with tab3:
        st.subheader("知识库统计")

        if st.button("📊 获取统计信息"):
            with st.spinner("正在获取统计信息..."):
                try:
                    response = asyncio.run(
                        st.session_state.main_agent.chat("请获取知识库的统计信息")
                    )
                    st.success("✅ 统计信息获取成功")
                    st.write(response.content)

                except Exception as e:
                    st.error(f"❌ 获取失败: {e}")


def render_workflow_interface():
    """渲染工作流程界面"""
    st.header("🔄 智能工作流程")

    if not st.session_state.main_agent:
        st.warning("请先初始化AI助手系统")
        return

    # 预定义工作流程
    workflow_type = st.selectbox(
        "选择工作流程",
        options=["research", "analysis", "planning"],
        format_func=lambda x: {
            "research": "🔬 研究工作流程",
            "analysis": "📊 分析工作流程",
            "planning": "📋 规划工作流程",
        }[x],
    )

    # 根据工作流程类型显示不同的参数输入
    if workflow_type == "research":
        st.subheader("🔬 研究工作流程")
        st.info("将自动执行：制定研究计划 → 搜索资料 → 整理知识库")

        query = st.text_input("研究主题", placeholder="例如：AI在教育中的应用")

    elif workflow_type == "analysis":
        st.subheader("📊 分析工作流程")
        st.info("将自动执行：检索相关知识 → 综合分析 → 生成报告")

        query = st.text_input("分析主题", placeholder="例如：当前项目的技术可行性")

    elif workflow_type == "planning":
        st.subheader("📋 规划工作流程")
        st.info("将自动执行：制定详细计划 → 风险评估 → 资源分析")

        query = st.text_input("规划目标", placeholder="例如：开发AI教育助手原型")

    if st.button("🚀 执行工作流程") and query:
        with st.spinner(f"正在执行{workflow_type}工作流程..."):
            try:
                # 执行工作流程
                parameters = (
                    {"query": query}
                    if workflow_type == "research"
                    else (
                        {"topic": query}
                        if workflow_type == "analysis"
                        else {"goal": query}
                    )
                )

                responses = asyncio.run(
                    st.session_state.main_agent.execute_workflow(
                        workflow_type, parameters
                    )
                )

                st.success(f"✅ {workflow_type}工作流程执行完成！")

                # 显示每个步骤的结果
                for i, response in enumerate(responses, 1):
                    with st.expander(f"步骤 {i}: {response.agent_role.value}"):
                        st.write(response.content)
                        st.caption(f"置信度: {response.confidence:.2f}")

            except Exception as e:
                st.error(f"❌ 工作流程执行失败: {e}")


def render_chat_interface():
    """渲染智能对话界面"""
    st.header("💬 智能对话")

    # 加载Agent
    if not load_agent():
        st.error("无法加载AI助手，请检查配置。")
        return

    # 状态显示区域 - 显示实时日志
    st.subheader("🔍 AI助手状态")
    status_container = st.empty()
    progress_container = st.progress(0)

    # 将UI更新器设置到代理
    ui_updater = StreamlitStatusUpdater(status_container, progress_container)
    if st.session_state.main_agent:
        st.session_state.main_agent.set_ui_updater(ui_updater)

    # 初始显示最新日志
    ui_updater.update_with_latest_logs()

    # 显示聊天记录
    st.subheader("💬 对话记录")
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("请输入您的问题或指令..."):
        # 记录用户输入
        info(
            "用户发送消息", user_id=st.session_state.user_id, message_length=len(prompt)
        )

        # 添加用户消息到历史记录
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用Agent处理
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                info(
                    "开始处理用户请求",
                    request=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                )

                # 使用带监控的chat方法
                response = asyncio.run(
                    st.session_state.main_agent.chat_with_monitoring(
                        prompt, session_id=st.session_state.user_id
                    )
                )

                full_response = response.content
                message_placeholder.markdown(full_response)

                # 添加助手响应到历史记录
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )

                # 保存状态
                save_user_state_to_storage()

                success("用户请求处理完成", response_length=len(full_response))

            except Exception as e:
                error("处理用户请求失败", error=str(e))
                st.error(f"处理请求时出错: {e}")
                full_response = f"抱歉，处理您的请求时遇到了错误: {e}"
                message_placeholder.error(full_response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )

    # 实时日志面板
    if st.session_state.app_settings.get("show_status_history", False):
        with st.expander("🔍 查看详细日志", expanded=False):
            if st.session_state.log_messages:
                st.subheader("📋 系统日志")
                # 显示最近20条日志
                recent_logs = st.session_state.log_messages[-20:]
                for log_entry in reversed(recent_logs):
                    timestamp = log_entry["timestamp"]
                    level = log_entry["level"]
                    message = log_entry["message"]

                    # 根据级别选择颜色
                    if level == "ERROR":
                        st.error(f"[{timestamp}] {message}")
                    elif level == "SUCCESS":
                        st.success(f"[{timestamp}] {message}")
                    elif level == "WARNING":
                        st.warning(f"[{timestamp}] {message}")
                    else:
                        st.info(f"[{timestamp}] {message}")
            else:
                st.info("暂无日志记录。")

            # AI执行历史
            if st.session_state.main_agent:
                st.subheader("🤖 AI执行历史")
                status_history = st.session_state.main_agent.get_status_history()
                if status_history:
                    for status in reversed(status_history):
                        st.text(
                            f"[{status['timestamp']}] {status['emoji']} {status['status']}"
                        )
                else:
                    st.info("暂无AI执行记录。")


def main():
    """主函数"""
    # 初始化
    init_session_state()

    # 渲染页面
    render_header()
    render_sidebar()

    # 创建主要布局：左侧内容区域，右侧日志面板
    main_col = st.columns([1])[0]

    with main_col:
        # 主要内容区域
        main_tabs = st.tabs(
            [
                "💬 智能对话",
                "🚀 项目中心",
            ]
        )

        with main_tabs[0]:
            render_chat_interface()

        with main_tabs[1]:
            st.header("🚀 项目中心")
            hub_tabs = st.tabs(["📋 项目详情", "📝 任务管理", "📚 知识库"])
            with hub_tabs[0]:
                render_project_management()
            with hub_tabs[1]:
                render_task_management()
            with hub_tabs[2]:
                render_knowledge_management()

    # 页脚
    st.divider()
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚀 CreatPartner v1.1.0 - AI驱动的创新创业项目助手</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
