#!/usr/bin/env python3
"""
测试增强的搜索代理 - 基于Jina AI完整生态
验证所有Jina AI服务的集成效果
"""

import asyncio
import os
from dotenv import load_dotenv
from search_agent import create_search_agent, create_search_dependencies

# 加载环境变量
load_dotenv()

async def test_jina_search():
    """测试Jina Search API"""
    print("\n🔍 测试Jina Search API...")
    
    agent = create_search_agent()
    deps = create_search_dependencies(max_results=3)
    
    try:
        result = await agent.search(
            "人工智能在创新创业中的应用", 
            deps, 
            "jina_search"
        )
        print("✅ Jina Search测试成功")
        print(f"📊 结果: {result[:300]}...")
    except Exception as e:
        print(f"❌ Jina Search测试失败: {e}")

async def test_arxiv_search():
    """测试Arxiv搜索"""
    print("\n📚 测试Arxiv学术搜索...")
    
    agent = create_search_agent()
    deps = create_search_dependencies(max_results=2)
    
    try:
        result = await agent.search(
            "artificial intelligence startup", 
            deps, 
            "arxiv"
        )
        print("✅ Arxiv搜索测试成功")
        print(f"📊 结果: {result[:300]}...")
    except Exception as e:
        print(f"❌ Arxiv搜索测试失败: {e}")

async def test_jina_reader():
    """测试Jina Reader API"""
    print("\n📄 测试Jina Reader API...")
    
    agent = create_search_agent()
    deps = create_search_dependencies()
    
    test_urls = [
        "https://jina.ai",
        "https://example.com"
    ]
    
    try:
        result = await agent.extract_content(test_urls, deps, analyze=False)
        print("✅ Jina Reader测试成功")
        print(f"📊 提取结果数量: {len(result.get('results', []))}")
    except Exception as e:
        print(f"❌ Jina Reader测试失败: {e}")

async def test_comprehensive_search():
    """测试综合搜索功能"""
    print("\n🌟 测试综合搜索...")
    
    agent = create_search_agent()
    deps = create_search_dependencies(max_results=2)
    
    try:
        result = await agent.search(
            "区块链技术在供应链管理中的创新应用", 
            deps, 
            "comprehensive"
        )
        print("✅ 综合搜索测试成功")
        print(f"📊 结果: {result[:400]}...")
    except Exception as e:
        print(f"❌ 综合搜索测试失败: {e}")

async def test_deep_search():
    """测试DeepSearch功能"""
    print("\n🧠 测试DeepSearch...")
    
    # 检查是否有API密钥
    if not os.getenv("JINA_API_KEY"):
        print("⚠️ 跳过DeepSearch测试 - 需要JINA_API_KEY")
        return
    
    agent = create_search_agent()
    deps = create_search_dependencies(enable_deep_search=True)
    
    try:
        result = await agent.search(
            "分析智能制造在中小企业中的应用前景和挑战", 
            deps, 
            "deepsearch"
        )
        print("✅ DeepSearch测试成功")
        print(f"📊 结果: {result[:400]}...")
    except Exception as e:
        print(f"❌ DeepSearch测试失败: {e}")

async def test_search_agent_tools():
    """测试搜索代理的所有工具"""
    print("\n🛠️ 测试搜索代理工具...")
    
    agent = create_search_agent()
    deps = create_search_dependencies(max_results=2)
    
    # 测试工具是否正确注册
    tools = []
    if hasattr(agent.agent, '_tools'):
        tools = list(agent.agent._tools.keys())
    
    expected_tools = [
        "jina_search", 
        "jina_reader", 
        "jina_deepsearch", 
        "jina_classify",
        "arxiv_search", 
        "comprehensive_search", 
        "extract_and_analyze_urls"
    ]
    
    print(f"📋 注册的工具: {tools}")
    
    for tool in expected_tools:
        if tool in str(tools):
            print(f"✅ {tool} - 已注册")
        else:
            print(f"❌ {tool} - 未找到")

def test_search_dependencies():
    """测试搜索依赖配置"""
    print("\n⚙️ 测试搜索依赖配置...")
    
    # 测试默认配置
    deps1 = create_search_dependencies()
    print(f"✅ 默认配置: max_results={deps1.max_results}")
    
    # 测试自定义配置
    deps2 = create_search_dependencies(
        max_results=10,
        enable_deep_search=True,
        enable_content_extraction=True
    )
    print(f"✅ 自定义配置: max_results={deps2.max_results}, deep_search={deps2.enable_deep_search}")
    
    # 测试环境变量
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        print(f"✅ JINA_API_KEY: {'已配置' if jina_key else '未配置'}")
    else:
        print("⚠️ JINA_API_KEY: 未配置 - 某些功能可能受限")

async def main():
    """主测试函数"""
    print("🧪 增强搜索代理测试套件")
    print("=" * 60)
    
    # 环境检查
    print("\n🔧 环境检查...")
    required_packages = ["httpx", "arxiv", "pydantic_ai"]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
    
    # 运行所有测试
    test_search_dependencies()
    await test_search_agent_tools()
    await test_jina_search()
    await test_arxiv_search()
    await test_jina_reader()
    await test_comprehensive_search()
    await test_deep_search()
    
    print("\n🎉 搜索代理测试完成！")
    print("\n📖 功能特点:")
    print("   1. Jina Search API - 高质量网络搜索")
    print("   2. Jina Reader API - 智能网页内容提取")
    print("   3. Jina DeepSearch API - 深度研究和推理")
    print("   4. Jina Classifier API - 智能内容分类")
    print("   5. Arxiv API - 学术论文搜索")
    print("   6. 综合搜索 - 多源信息整合")
    print("   7. 内容分析 - URL提取和智能分析")

if __name__ == "__main__":
    asyncio.run(main())
