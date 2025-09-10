#!/usr/bin/env python3
"""
测试搜索代理功能
基于现有项目架构，专注测试Jina AI搜索集成
"""

import asyncio
from config import config, validate_config
from search_agent import SearchAgent, create_search_dependencies

def print_search_test_banner():
    """打印搜索测试横幅"""
    print("🔍 CreatPartner 搜索代理测试")
    print("=" * 50)

async def test_basic_search():
    """测试基本搜索功能"""
    print("\n📝 测试基本搜索功能...")
    
    # 创建搜索代理
    search_agent = SearchAgent()
    
    # 创建搜索依赖
    search_deps = create_search_dependencies(
        max_results=3,
        enable_deep_search=False,
        enable_content_extraction=True
    )
    
    # 测试查询列表
    test_queries = [
        "人工智能在教育中的应用",
        "machine learning education trends",
        "创新创业项目案例分析"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 搜索 {i}: {query}")
        try:
            result = await search_agent.search(
                query=query,
                deps=search_deps,
                search_type="jina_search"
            )
            print(f"✅ 搜索成功")
            print(f"📄 结果摘要: {result[:200]}...")
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")

async def test_arxiv_search():
    """测试学术论文搜索"""
    print("\n📚 测试学术论文搜索...")
    
    search_agent = SearchAgent()
    search_deps = create_search_dependencies(max_results=2)
    
    academic_queries = [
        "artificial intelligence education",
        "machine learning pedagogy",
        "computer science learning"
    ]
    
    for i, query in enumerate(academic_queries, 1):
        print(f"\n📖 学术搜索 {i}: {query}")
        try:
            result = await search_agent.search(
                query=query,
                deps=search_deps,
                search_type="arxiv"
            )
            print(f"✅ 学术搜索成功")
            print(f"📄 结果摘要: {result[:200]}...")
            
        except Exception as e:
            print(f"❌ 学术搜索失败: {e}")

async def test_comprehensive_search():
    """测试综合搜索（网络+学术）"""
    print("\n🌐 测试综合搜索...")
    
    search_agent = SearchAgent()
    search_deps = create_search_dependencies(
        max_results=2,
        enable_content_extraction=True
    )
    
    comprehensive_queries = [
        "AI驱动的个性化学习平台",
        "智能教育技术发展趋势"
    ]
    
    for i, query in enumerate(comprehensive_queries, 1):
        print(f"\n🔄 综合搜索 {i}: {query}")
        try:
            result = await search_agent.search(
                query=query,
                deps=search_deps,
                search_type="comprehensive"
            )
            print(f"✅ 综合搜索成功")
            print(f"📄 结果摘要: {result[:200]}...")
            
        except Exception as e:
            print(f"❌ 综合搜索失败: {e}")

async def test_jina_services_individually():
    """单独测试Jina服务组件"""
    print("\n🧪 单独测试Jina服务组件...")
    
    if not config.embedding.api_key:
        print("❌ 缺少Jina API密钥，跳过服务测试")
        return
    
    from search_agent import JinaEmbeddingService, JinaRerankerService, JinaSegmenterService
    
    # 测试嵌入服务
    print("\n🔤 测试嵌入服务...")
    try:
        embedding_service = JinaEmbeddingService(
            config.embedding.api_key, 
            config.embedding.model
        )
        
        test_text = "这是一个测试文本，用于验证嵌入服务功能"
        embedding = await embedding_service.get_single_embedding(test_text)
        
        print(f"✅ 嵌入生成成功")
        print(f"   维度: {len(embedding)}")
        print(f"   前5个值: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ 嵌入服务测试失败: {e}")
    
    # 测试重排序服务
    print("\n🔀 测试重排序服务...")
    try:
        reranker_service = JinaRerankerService(config.embedding.api_key)
        
        query = "如何提高学习效率"
        documents = [
            "制定明确的学习计划和目标",
            "使用番茄工作法管理时间",
            "创造良好的学习环境",
            "定期复习和总结知识点"
        ]
        
        reranked_results = await reranker_service.rerank(query, documents, top_n=3)
        
        print(f"✅ 重排序成功")
        print(f"   查询: {query}")
        print(f"   重排序后前3个结果:")
        for i, result in enumerate(reranked_results[:3]):
            score = result.get('relevance_score', 0)
            doc_text = result.get('document', {}).get('text', '未知文档')
            print(f"     {i+1}. 分数: {score:.4f} - {doc_text}")
        
    except Exception as e:
        print(f"❌ 重排序服务测试失败: {e}")
    
    # 测试文本分割服务
    print("\n✂️ 测试文本分割服务...")
    try:
        segmenter_service = JinaSegmenterService(config.embedding.api_key)
        
        long_text = """
        创新创业教育是高等教育改革的重要内容，旨在培养学生的创新精神、创业意识和创新创业能力。
        通过系统的课程体系、实践平台和指导服务，帮助学生理解创新创业的基本理论和方法。
        创新创业项目是实践教学的重要载体，包括创新训练项目、创业训练项目和创业实践项目。
        学生通过参与项目，可以锻炼问题发现、方案设计、团队协作、资源整合等综合能力。
        项目实施过程中需要重视市场调研、技术可行性分析、商业模式设计和风险评估等关键环节。
        """ * 3  # 重复内容创建更长的文本
        
        chunks = await segmenter_service.segment_text(long_text, max_chunk_length=300)
        
        print(f"✅ 文本分割成功")
        print(f"   原文长度: {len(long_text)} 字符")
        print(f"   分割成 {len(chunks)} 个片段")
        for i, chunk in enumerate(chunks):
            print(f"     片段 {i+1}: {len(chunk)} 字符")
        
    except Exception as e:
        print(f"❌ 文本分割服务测试失败: {e}")

def test_search_dependencies():
    """测试搜索依赖配置"""
    print("\n⚙️ 测试搜索依赖配置...")
    
    # 测试默认配置
    deps1 = create_search_dependencies()
    print(f"✅ 默认配置:")
    print(f"   最大结果数: {deps1.max_results}")
    print(f"   启用深度搜索: {deps1.enable_deep_search}")
    print(f"   API密钥已配置: {'是' if deps1.jina_api_key else '否'}")
    
    # 测试自定义配置
    deps2 = create_search_dependencies(
        max_results=10,
        enable_deep_search=True,
        enable_content_extraction=False
    )
    print(f"✅ 自定义配置:")
    print(f"   最大结果数: {deps2.max_results}")
    print(f"   启用深度搜索: {deps2.enable_deep_search}")
    print(f"   启用内容提取: {deps2.enable_content_extraction}")

async def main():
    """主测试函数"""
    print_search_test_banner()
    
    # 检查配置
    if not validate_config():
        print("⚠️ 配置验证失败，某些测试可能无法运行")
        print("建议检查 .env 文件中的API密钥配置")
    
    # 运行测试
    test_search_dependencies()
    
    if config.embedding.api_key:
        await test_jina_services_individually()
        await test_basic_search()
        await test_arxiv_search()
        await test_comprehensive_search()
    else:
        print("\n⚠️ 缺少Jina API密钥，跳过在线搜索测试")
        print("请在 .env 文件中配置 JINA_API_KEY")
    
    print("\n🎉 搜索代理测试完成！")
    print("\n📖 搜索功能特点:")
    print("   • 🔍 多源搜索：网络搜索 + 学术论文")
    print("   • 🧠 智能处理：嵌入向量 + 重排序")
    print("   • ✂️ 文本分割：长文档智能分块")
    print("   • 🎯 结果优化：相关性排序")
    print("   • 🔧 灵活配置：可调整搜索参数")

if __name__ == "__main__":
    asyncio.run(main())
