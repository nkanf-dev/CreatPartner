"""
CreatPartner Streamlit Web应用
基于新的多代理架构和配置系统提供友好的Web界面
"""

import streamlit as st
import asyncio
from datetime import datetime
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from config import config, validate_config
    from main_agent import CreatPartnerAgent, SharedDependencies
    from search_agent import SearchAgent, SearchDependencies
    from knowledge_agent import KnowledgeAgent, KnowledgeDependencies
    
except ImportError as e:
    st.error(f"导入错误: {e}")
    st.error("请先运行: python install.py")
    st.stop()


def init_session_state():
    """初始化会话状态"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'project_name' not in st.session_state:
        st.session_state.project_name = ""
    if 'project_description' not in st.session_state:
        st.session_state.project_description = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'knowledge_stats' not in st.session_state:
        st.session_state.knowledge_stats = {}
    if 'quick_prompt' not in st.session_state:
        st.session_state.quick_prompt = None


def check_environment():
    """检查环境配置"""
    missing_vars = []
    warnings = []
    
    # 检查LLM配置
    if not config.llm.api_key:
        missing_vars.append("LLM API密钥 (SILICONFLOW_API_KEY)")
    
    # 检查数据库配置
    if not config.database.mongodb_uri or config.database.mongodb_uri == "mongodb://localhost:27017":
        warnings.append("使用默认MongoDB配置")
    
    # 检查嵌入服务（可选）
    if not config.embedding.api_key:
        warnings.append("未配置Jina API密钥，某些功能可能受限")
    
    return missing_vars, warnings


def main():
    """主应用"""
    st.set_page_config(
        page_title="CreatPartner - AI创新助手",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化
    init_session_state()
    
    # 页面标题
    st.title("🚀 CreatPartner")
    st.subheader("AI驱动的创新竞赛助手")
    
    # 检查环境
    missing_vars, warnings = check_environment()
    
    if missing_vars:
        st.error("❌ 缺少必要配置:")
        for var in missing_vars:
            st.error(f"  • {var}")
        st.info("请在 .env 文件中配置必要的API密钥")
        
        with st.expander("📖 配置说明"):
            st.code("""
# 在 .env 文件中添加以下配置:
SILICONFLOW_API_KEY=你的硅基流动API密钥
MONGODB_URI=mongodb://localhost:27017  # 可选，默认本地MongoDB
JINA_API_KEY=你的Jina API密钥  # 可选，用于增强搜索功能
DB_NAME=creatpartner  # 可选，数据库名称
            """)
        st.stop()
    
    if warnings:
        with st.expander("⚠️ 配置警告"):
            for warning in warnings:
                st.warning(f"  • {warning}")
    
    # 验证完整配置
    if not validate_config():
        st.warning("某些功能可能因配置不完整而受限")
    
    # 侧边栏
    with st.sidebar:
        st.header("🎯 项目配置")
        
        # 项目设置
        project_name = st.text_input(
            "项目名称",
            value=st.session_state.project_name,
            placeholder="输入你的项目名称"
        )
        
        project_description = st.text_area(
            "项目描述",
            value=st.session_state.project_description,
            placeholder="简要描述你的项目..."
        )
        
        if st.button("🚀 开始研究会话"):
            if project_name:
                st.session_state.project_name = project_name
                st.session_state.project_description = project_description
                
                # 创建主代理
                st.session_state.agent = CreatPartnerAgent()
                
                # 创建项目会话依赖
                shared_deps = SharedDependencies(
                    project_name=project_name,
                    project_description=project_description,
                    project_stage="research",
                    jina_api_key=config.embedding.api_key,
                    mongodb_uri=config.database.mongodb_uri,
                    database_name=config.database.database_name,
                    max_search_results=config.search.max_results
                )
                
                # 开始会话
                with st.spinner("初始化AI助手..."):
                    try:
                        # 使用新的配置系统异步启动会话
                        welcome = asyncio.run(
                            st.session_state.agent.start_project_session(
                                project_name, project_description, "research"
                            )
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": welcome,
                            "timestamp": datetime.now()
                        })
                        st.success("研究会话已开始！")
                    except Exception as e:
                        st.error(f"初始化失败: {e}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.warning("请输入项目名称")
        
        # 功能选项
        st.header("🔧 功能选项")
        
        search_only = st.checkbox("仅搜索模式", help="只使用搜索功能，不存储到知识库")
        
        max_results = st.slider("最大搜索结果", 1, 10, 5)
        
        # 系统状态
        st.header("📊 系统状态")
        
        if st.button("🔍 检查状态"):
            with st.spinner("检查中..."):
                status = {}
                
                # 检查LLM配置
                if config.llm.api_key:
                    status["LLM服务"] = f"✅ {config.llm.provider}"
                else:
                    status["LLM服务"] = "❌ 未配置API密钥"
                
                # 检查搜索代理
                try:
                    search_deps = SearchDependencies(
                        jina_api_key=config.embedding.api_key,
                        max_results=config.search.max_results
                    )
                    status["搜索代理"] = "✅ 正常"
                except Exception as e:
                    status["搜索代理"] = f"❌ {str(e)[:50]}"
                
                # 检查MongoDB连接
                try:
                    import pymongo
                    client = pymongo.MongoClient(
                        config.database.mongodb_uri,
                        serverSelectionTimeoutMS=2000
                    )
                    client.server_info()
                    status["MongoDB"] = "✅ 已连接"
                    client.close()
                except Exception as e:
                    status["MongoDB"] = f"❌ {str(e)[:50]}"
                
                # 检查知识库
                if config.embedding.api_key:
                    status["知识库"] = "✅ Jina已配置"
                else:
                    status["知识库"] = "⚠️ 基础功能可用"
                
                for name, stat in status.items():
                    st.text(f"{name}: {stat}")
    
    # 主内容区域
    if st.session_state.agent is None:
        st.info("👈 请在侧边栏配置项目并开始会话")
        
        # 显示功能介绍
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔍 智能搜索")
            st.write("- Web搜索 (Jina)")
            st.write("- 学术搜索 (Arxiv)")
            st.write("- 智能分析总结")
        
        with col2:
            st.subheader("📚 知识管理")
            st.write("- 项目长期记忆")
            st.write("- 外部资料库")
            st.write("- 向量语义搜索")
        
        with col3:
            st.subheader("🧠 项目分析")
            st.write("- 技术可行性评估")
            st.write("- 市场竞争分析")
            st.write("- 创新点识别")
    
    else:
        # 聊天界面
        st.header(f"💬 与 {st.session_state.project_name} 对话")
        
        # 显示聊天历史
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                st.caption(f"时间: {msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 用户输入
        if prompt := st.chat_input("请输入你的问题或需求..."):
            # 添加用户消息
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })
            
            # 显示用户消息
            with st.chat_message("user"):
                st.write(prompt)
            
            # 生成回复
            with st.chat_message("assistant"):
                with st.spinner("AI正在思考..."):
                    try:
                        # 创建共享依赖
                        deps = SharedDependencies(
                            project_name=st.session_state.project_name,
                            project_description=st.session_state.project_description,
                            project_stage="research",
                            jina_api_key=config.embedding.api_key,
                            mongodb_uri=config.database.mongodb_uri,
                            database_name=config.database.database_name,
                            max_search_results=max_results
                        )
                        
                        # 使用新的多代理架构
                        response = asyncio.run(
                            st.session_state.agent.agent.run(prompt, deps=deps)
                        )
                        
                        # 处理响应
                        response_text = response.output if hasattr(response, 'output') else str(response)
                        st.write(response_text)
                        
                        # 添加助手回复到历史
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "timestamp": datetime.now()
                        })
                        
                    except Exception as e:
                        error_msg = f"抱歉，处理您的请求时出现错误: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now()
                        })
                        import traceback
                        st.error(traceback.format_exc())
        
        # 快捷操作按钮
        st.header("⚡ 快捷操作")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 市场分析"):
                quick_prompt = f"请分析{st.session_state.project_name}项目的市场情况和竞争态势"
                st.session_state.quick_prompt = quick_prompt
                st.rerun()
        
        with col2:
            if st.button("🎯 技术评估"):
                quick_prompt = f"请评估{st.session_state.project_name}项目的技术可行性"
                st.session_state.quick_prompt = quick_prompt
                st.rerun()
        
        with col3:
            if st.button("💡 创新建议"):
                quick_prompt = f"请为{st.session_state.project_name}项目提供创新建议"
                st.session_state.quick_prompt = quick_prompt
                st.rerun()
        
        with col4:
            if st.button("📊 项目总结"):
                quick_prompt = f"请总结{st.session_state.project_name}项目的当前状况"
                st.session_state.quick_prompt = quick_prompt
                st.rerun()
        
        # 处理快捷操作
        if hasattr(st.session_state, 'quick_prompt') and st.session_state.quick_prompt:
            prompt = st.session_state.quick_prompt
            st.session_state.quick_prompt = None  # 清除快捷提示
            
            # 添加用户消息
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })
            
            # 生成回复
            with st.spinner("AI正在分析..."):
                try:
                    # 创建共享依赖
                    deps = SharedDependencies(
                        project_name=st.session_state.project_name,
                        project_description=st.session_state.project_description,
                        project_stage="research",
                        jina_api_key=config.embedding.api_key,
                        mongodb_uri=config.database.mongodb_uri,
                        database_name=config.database.database_name,
                        max_search_results=max_results
                    )
                    
                    # 使用多代理架构
                    response = asyncio.run(
                        st.session_state.agent.agent.run(prompt, deps=deps)
                    )
                    
                    # 处理响应
                    response_text = response.output if hasattr(response, 'output') else str(response)
                    
                    # 添加助手回复到历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now()
                    })
                    
                    # 重新运行显示结果
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"抱歉，处理快捷操作时出现错误: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
        
        # 项目状态面板
        with st.expander("📈 项目状态"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("对话轮数", len(st.session_state.chat_history))
                st.metric("项目阶段", "研究阶段")
            
            with col2:
                if st.session_state.chat_history:
                    last_time = st.session_state.chat_history[-1]["timestamp"]
                    st.metric("最后活动", last_time.strftime("%H:%M"))
                
                # 显示配置状态
                config_status = "✅ 完整" if config.llm.api_key and config.embedding.api_key else "⚠️ 部分"
                st.metric("配置状态", config_status)
    
    # 页脚
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("🚀 **CreatPartner** - 让AI助力你的创新创业之路！")
        st.caption(f"当前配置: {config.llm.provider.upper()} + MongoDB + Jina AI")


if __name__ == "__main__":
    main()
