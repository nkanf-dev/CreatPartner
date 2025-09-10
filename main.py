"""
CreatPartner - AI驱动的创新竞赛助手
完整的智能代理系统，包含搜索、知识管理和项目分析功能
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))


from main_agent import create_creatpartner_agent, MainAgentDependencies
from search_agent import SearchAgent, SearchDependencies
from knowledge_agent import KnowledgeAgent, KnowledgeDependencies
from dotenv import load_dotenv
    
load_dotenv()


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🚀 CreatPartner - AI创新竞赛助手                           ║
║                                                              ║
║    ✨ 功能特性:                                               ║
║    • 🔍 智能搜索 (Web + 学术论文)                             ║
║    • 📚 知识库管理 (项目记忆 + 外部资料)                       ║
║    • 🧠 项目分析和洞察                                        ║
║    • 💡 创新建议和方案                                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """检查环境配置"""
    print("🔧 检查环境配置...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API密钥 (必需)",
        "MONGODB_URI": "MongoDB连接字符串",
        "JINA_API_KEY": "Jina搜索API密钥 (可选)"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if var == "OPENAI_API_KEY" and not value:
            missing_vars.append(f"  • {var}: {description}")
        elif value:
            # 隐藏API密钥的显示
            display_value = value[:8] + "..." if len(value) > 8 else "已设置"
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ⚠️  {var}: 未设置 ({description})")
    
    if missing_vars:
        print("\n❌ 缺少必需的环境变量:")
        for var in missing_vars:
            print(var)
        print("\n请编辑 .env 文件添加必需的配置。")
        return False
    
    print("✅ 环境配置检查完成")
    return True


def test_individual_agents():
    """测试各个代理组件"""
    print("\n🧪 测试代理组件...")
    
    # 测试搜索代理
    print("\n1. 测试搜索代理...")
    try:
        search_agent = SearchAgent()
        search_deps = SearchDependencies(max_results=2)
        result = search_agent.search_sync("AI教育应用", search_deps)
        print(f"   ✅ 搜索代理工作正常")
        print(f"   📄 搜索结果预览: {result[:100]}...")
    except Exception as e:
        print(f"   ❌ 搜索代理测试失败: {e}")
    
    # 测试知识库代理
    print("\n2. 测试知识库代理...")
    try:
        knowledge_agent = KnowledgeAgent()
        knowledge_deps = KnowledgeDependencies(
            mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            database_name="creatpartner_test"
        )
        
        # 创建索引
        knowledge_agent.create_vector_search_index(knowledge_deps)
        print(f"   ✅ 知识库代理工作正常")
    except Exception as e:
        print(f"   ❌ 知识库代理测试失败: {e}")


def demo_main_agent():
    """演示主代理功能"""
    print("\n🎯 主代理功能演示...")
    
    try:
        # 创建主代理
        agent = create_creatpartner_agent()
        
        # 开始演示会话
        project_name = "智能学习助手"
        project_description = "基于AI的个性化教育平台，帮助学生提高学习效率"
        
        print(f"\n项目名称: {project_name}")
        print(f"项目描述: {project_description}")
        print("\n" + "="*60)
        
        # 同步演示
        welcome = agent.start_research_session_sync(
            project_name=project_name,
            project_description=project_description
        )
        
        print("🤖 CreatPartner 响应:")
        print(welcome)
        
        return agent, project_name, project_description
        
    except Exception as e:
        print(f"❌ 主代理演示失败: {e}")
        return None, None, None


async def interactive_demo(agent, project_name, project_description):
    """交互式演示"""
    print("\n💬 进入交互式演示模式...")
    print("输入 'quit' 退出演示")
    
    deps = MainAgentDependencies(
        project_name=project_name,
        project_description=project_description,
        jina_api_key=os.getenv("JINA_API_KEY"),
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    )
    
    demo_queries = [
        "请分析AI教育市场的发展趋势",
        "我们的项目有哪些技术难点？",
        "推荐一些类似的成功案例",
        "制定项目下一阶段的计划"
    ]
    
    print("\n🎯 建议的演示查询:")
    for i, query in enumerate(demo_queries, 1):
        print(f"  {i}. {query}")
    
    while True:
        try:
            user_input = input("\n👤 您的问题 (或输入数字选择建议查询): ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(demo_queries):
                    user_input = demo_queries[idx]
                else:
                    print("❌ 无效的选择")
                    continue
            
            if not user_input:
                continue
            
            print(f"\n🤔 处理中: {user_input}")
            print("⏳ 正在搜索和分析...")
            
            # 执行查询
            result = await agent.agent.run(user_input, deps=deps)
            
            print(f"\n🤖 CreatPartner 回复:")
            print("-" * 60)
            print(result.output)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 处理出错: {e}")


def main():
    """主程序入口"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        return
    
    # 测试组件
    test_individual_agents()
    
    # 演示主代理
    agent, project_name, project_description = demo_main_agent()
    
    if agent is None:
        print("❌ 无法启动主代理，请检查配置")
        return
    
    # 询问是否进入交互模式
    print("\n" + "="*60)
    choice = input("是否进入交互式演示？(y/N): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        try:
            asyncio.run(interactive_demo(agent, project_name, project_description))
        except KeyboardInterrupt:
            print("\n👋 演示结束")
    
    print("\n🎉 CreatPartner 演示完成!")
    print("\n📖 更多使用方法请查看 README.md")


async def async_main():
    """异步主程序 - 完整功能演示"""
    print_banner()
    
    # 创建主代理
    agent = create_creatpartner_agent()
    
    # 开始研究会话
    project_name = "绿色能源管理系统"
    project_description = "基于物联网和AI的智能能源监控与优化平台"
    
    welcome = await agent.start_research_session(
        project_name=project_name,
        project_description=project_description
    )
    print("🤖 CreatPartner:")
    print(welcome)
    
    # 执行研究任务
    deps = MainAgentDependencies(
        project_name=project_name,
        project_description=project_description,
        jina_api_key=os.getenv("JINA_API_KEY"),
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    )
    
    tasks = [
        "分析可再生能源管理的技术趋势",
        "研究物联网在能源监控中的应用",
        "评估我们项目的市场竞争力"
    ]
    
    for task in tasks:
        print(f"\n🔍 执行任务: {task}")
        result = await agent.agent.run(task, deps=deps)
        print(f"📋 结果: {result.output[:200]}...")


if __name__ == "__main__":
    # 运行同步版本（默认）
    main()
    
    # 可选：运行异步完整演示
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        print("\n" + "="*60)
        print("🚀 运行异步完整演示...")
        asyncio.run(async_main())
