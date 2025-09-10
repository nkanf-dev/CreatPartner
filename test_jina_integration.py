#!/usr/bin/env python3
"""
测试Jina AI API集成的脚本
验证知识代理使用Jina服务的功能
"""

import asyncio
from config import config
from knowledge_agent import KnowledgeAgent, KnowledgeDependencies

async def test_jina_services():
    """测试Jina AI服务集成"""
    
    # 检查环境变量
    jina_api_key = config.embedding.api_key
    if not jina_api_key:
        print("❌ 缺少JINA_API_KEY配置")
        return
    
    print("🚀 开始测试Jina AI服务集成...")
    
    # 创建知识代理
    knowledge_agent = KnowledgeAgent()
    
    # 创建依赖配置
    deps = KnowledgeDependencies(
        mongodb_uri=config.database.mongodb_uri,
        database_name="creatpartner_test",
        jina_api_key=jina_api_key,
        embedding_model="jina-embeddings-v3"
    )
    
    try:
        # 测试1: 添加知识条目（自动分割长文本）
        print("\n📝 测试1: 添加知识条目...")
        test_content = """
        大学生创新创业项目是培养学生创新精神和实践能力的重要平台。
        项目通常分为创新训练项目、创业训练项目和创业实践项目三个层次。
        创新训练项目重点培养学生的创新思维和科研能力，要求学生在导师指导下独立完成创新性研究。
        创业训练项目则注重商业模式的设计和市场调研，帮助学生了解创业的基本流程。
        创业实践项目要求学生实际注册公司并运营，是最高层次的创业教育实践。
        成功的项目往往具有明确的创新点、可行的技术方案、清晰的商业模式和完整的团队配置。
        """
        
        result = await knowledge_agent.agent.run(
            f"请添加以下内容到项目记忆知识库：标题是'大学生创新创业项目指南'，内容是：{test_content}，来源是'教育部文件'，标签是['创新创业', '项目指南', '大学生']",
            deps=deps
        )
        print(f"添加结果: {result.output}")
        
        # 测试2: 搜索知识（使用重排序）
        print("\n🔍 测试2: 搜索知识...")
        search_result = await knowledge_agent.agent.run(
            "搜索关于'创业项目'的相关知识，使用重排序优化结果",
            deps=deps
        )
        print(f"搜索结果: {search_result.output}")
        
        # 测试3: 获取知识库统计
        print("\n📊 测试3: 获取知识库统计...")
        stats_result = await knowledge_agent.agent.run(
            "获取知识库的统计信息",
            deps=deps
        )
        print(f"统计信息: {stats_result.output}")
        
        # 测试4: 创建向量搜索索引
        print("\n🗂️ 测试4: 创建向量搜索索引...")
        knowledge_agent.create_vector_search_index(deps)
        
        print("\n✅ 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

async def test_individual_services():
    """单独测试各个Jina服务"""
    
    jina_api_key = config.embedding.api_key
    if not jina_api_key:
        print("❌ 缺少JINA_API_KEY配置")
        return
    
    print("\n🧪 测试单个Jina服务...")
    
    from knowledge_agent import JinaEmbeddingService, JinaRerankerService, JinaSegmenterService
    
    # 测试嵌入服务
    print("\n🔤 测试嵌入服务...")
    embedding_service = JinaEmbeddingService(jina_api_key, "jina-embeddings-v3")
    try:
        embedding = await embedding_service.get_single_embedding("这是一个测试文本")
        print(f"嵌入维度: {len(embedding)}")
        print(f"嵌入前5个值: {embedding[:5]}")
    except Exception as e:
        print(f"嵌入服务测试失败: {e}")
    
    # 测试重排序服务
    print("\n🔀 测试重排序服务...")
    reranker_service = JinaRerankerService(jina_api_key)
    try:
        documents = [
            "创新创业项目需要明确的商业模式",
            "技术创新是项目成功的关键因素",
            "团队配置对项目发展至关重要"
        ]
        reranked = await reranker_service.rerank(
            "如何制定商业计划", 
            documents, 
            top_n=2
        )
        print(f"重排序结果: {reranked}")
    except Exception as e:
        print(f"重排序服务测试失败: {e}")
    
    # 测试分割服务
    print("\n✂️ 测试文本分割服务...")
    segmenter_service = JinaSegmenterService(jina_api_key)
    try:
        long_text = "这是一个很长的文本。" * 100  # 创建长文本
        chunks = await segmenter_service.segment_text(
            long_text, 
            max_chunk_length=200
        )
        print(f"分割成 {len(chunks)} 个片段")
        print(f"第一个片段长度: {len(chunks[0])}")
    except Exception as e:
        print(f"分割服务测试失败: {e}")

if __name__ == "__main__":
    print("🧪 Jina AI集成测试")
    print("=" * 50)
    
    # 运行服务测试
    asyncio.run(test_individual_services())
    
    # 运行集成测试
    asyncio.run(test_jina_services())
