#!/usr/bin/env python3
"""
测试重新设计的CreatPartner多代理系统
基于Pydantic AI最佳实践：智能体委托和程序化智能体交接
使用统一的配置系统
"""

import asyncio
from config import config, validate_config
from main_agent import create_creatpartner_agent, create_shared_dependencies

async def test_multi_agent_architecture():
    """测试多代理架构"""
    
    print("🚀 CreatPartner 多代理系统测试")
    print("=" * 60)
    
    # 1. 创建代理实例
    print("\n📦 创建代理实例...")
    agent = create_creatpartner_agent()
    print("✅ 主代理创建成功")
    print("   - 研究协调器 (ResearchCoordinator)")
    print("   - 用户交互代理 (InteractionAgent)")
    print("   - 搜索代理委托 (SearchAgent)")
    print("   - 知识代理委托 (KnowledgeAgent)")
    
    # 2. 测试项目会话初始化
    print("\n📋 测试项目会话初始化...")
    try:
        welcome = await agent.start_project_session(
            project_name="智能垃圾分类系统",
            project_description="基于计算机视觉和IoT技术的智能垃圾分类解决方案",
            project_stage="research"
        )
        print("✅ 项目会话初始化成功")
        print(f"🤖 系统欢迎信息:\n{welcome[:200]}...")
    except Exception as e:
        print(f"❌ 项目会话初始化失败: {e}")
        return
    
    # 3. 创建测试依赖配置
    print("\n⚙️ 创建依赖配置...")
    deps = create_shared_dependencies(
        project_name="智能垃圾分类系统",
        project_description="基于计算机视觉和IoT技术的智能垃圾分类解决方案",
        project_stage="research",
        request_limit=10,
        total_tokens_limit=5000,
        tool_calls_limit=5
    )
    print("✅ 依赖配置创建成功")
    print(f"   - 项目名称: {deps.project_name}")
    print(f"   - 项目阶段: {deps.project_stage}")
    print(f"   - 请求限制: {deps.usage_limits.request_limit}")
    
    # 4. 测试程序化智能体交接工作流
    print("\n🔄 测试程序化智能体交接工作流...")
    test_queries = [
        "当前垃圾分类技术的发展现状和挑战",
        "计算机视觉在垃圾识别中的应用案例",
        "IoT传感器在垃圾分类系统中的作用"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试查询 {i}: {query}")
        try:
            result = await agent.research_workflow(query, deps)
            print(f"✅ 工作流完成")
            print(f"📊 结果摘要: {result[:150]}...")
        except Exception as e:
            print(f"❌ 工作流失败: {e}")
    
    # 5. 测试智能体委托
    print("\n🤝 测试智能体委托...")
    
    # 直接与用户交互代理对话
    test_messages = [
        "我们项目的主要创新点应该是什么？",
        "请分析垃圾分类市场的竞争格局",
        "记录一个重要决策：我们决定重点关注厨余垃圾的智能识别"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n💬 对话 {i}: {message}")
        try:
            response = await agent.chat(message, deps)
            print(f"✅ 对话成功")
            print(f"🤖 回复摘要: {response[:150]}...")
        except Exception as e:
            print(f"❌ 对话失败: {e}")
    
    # 6. 测试使用统计
    print("\n📈 测试完成统计...")
    if hasattr(deps, 'usage_limits'):
        print(f"   - 配置的请求限制: {deps.usage_limits.request_limit}")
        print(f"   - 配置的令牌限制: {deps.usage_limits.total_tokens_limit}")
        print(f"   - 配置的工具调用限制: {deps.usage_limits.tool_calls_limit}")
    
    print("\n🎉 多代理架构测试完成！")

async def test_individual_agents():
    """测试单个代理的功能"""
    
    print("\n🧪 单代理功能测试")
    print("=" * 40)
    
    agent = create_creatpartner_agent()
    deps = create_shared_dependencies(
        project_name="测试项目",
        project_description="用于测试的项目",
        request_limit=5
    )
    
    # 测试研究协调器
    print("\n🎯 测试研究协调器...")
    try:
        coordinator_result = await agent.research_coordinator.coordinator.run(
            "测试协调器功能：分析垃圾分类技术趋势",
            deps=deps
        )
        print("✅ 研究协调器工作正常")
        print(f"📋 输出类型: {type(coordinator_result.output)}")
    except Exception as e:
        print(f"❌ 研究协调器测试失败: {e}")
    
    # 测试用户交互代理
    print("\n👤 测试用户交互代理...")
    try:
        interaction_result = await agent.interaction_agent.run(
            "你好，请介绍一下你的功能",
            deps=deps
        )
        print("✅ 用户交互代理工作正常")
        print(f"📋 输出类型: {type(interaction_result.output)}")
    except Exception as e:
        print(f"❌ 用户交互代理测试失败: {e}")

def test_dependency_configuration():
    """测试依赖配置"""
    
    print("\n⚙️ 依赖配置测试")
    print("=" * 30)
    
    # 测试默认配置
    deps1 = create_shared_dependencies("测试项目1")
    print(f"✅ 默认配置: {deps1.project_name}")
    
    # 测试自定义配置
    deps2 = create_shared_dependencies(
        "测试项目2",
        project_description="自定义描述",
        project_stage="development",
        max_search_results=10,
        request_limit=20
    )
    print(f"✅ 自定义配置: {deps2.project_name}, 阶段: {deps2.project_stage}")
    print(f"   搜索结果限制: {deps2.max_search_results}")
    print(f"   请求限制: {deps2.usage_limits.request_limit}")

async def main():
    """主测试函数"""
    
    # 检查配置有效性
    if not validate_config():
        print("⚠️ 配置验证失败")
        print("请检查 .env 文件中的API密钥配置")
        return
    
    print("🔧 配置检查通过")
    
    # 运行测试
    test_dependency_configuration()
    await test_individual_agents()
    await test_multi_agent_architecture()
    
    print("\n🎊 所有测试完成！")
    print("\n📖 多代理架构特点:")
    print("   1. 智能体委托 - 自动选择合适的子代理处理任务")
    print("   2. 程序化交接 - 结构化的工作流程")
    print("   3. 共享依赖 - 统一的配置管理")
    print("   4. 使用限制 - 防止过度调用和成本失控")
    print("   5. 错误恢复 - 优雅的错误处理机制")

if __name__ == "__main__":
    asyncio.run(main())
