"""
简化版知识库代理测试 - 不使用向量搜索
专注测试基本的CRUD功能和Jina API集成
基于现有项目架构重新构建
"""

import asyncio
from config import config, validate_config
from knowledge_agent import KnowledgeAgent, KnowledgeDependencies

def print_knowledge_test_banner():
    """打印知识库测试横幅"""
    print("📚 CreatPartner 知识库代理测试")
    print("=" * 50)

async def test_basic_knowledge_operations():
    """测试基本的知识库操作（不使用向量搜索）"""
    print("\n📝 测试基本知识库操作...")
    
    # 检查配置
    if not config.embedding.api_key:
        print("❌ 缺少JINA_API_KEY配置，跳过知识库测试")
        return
    
    print("🚀 开始测试基本知识库操作...")
    
    # 创建知识代理
    knowledge_agent = KnowledgeAgent()
    
    # 创建依赖配置
    deps = KnowledgeDependencies(
        mongodb_uri=config.database.mongodb_uri,
        database_name="creatpartner_test_simple",
        jina_api_key=config.embedding.api_key,
        embedding_model=config.embedding.model
    )
    
    try:
        # 测试1: 添加项目记忆知识
        print("\n📝 测试1: 添加项目记忆知识...")
        project_memory_content = "我们决定采用React前端框架和Node.js后端来开发教育平台，因为团队对这些技术比较熟悉。"
        
        result = await knowledge_agent.agent.run(
            f"使用add_knowledge工具添加项目记忆：标题='技术栈选择决策'，内容='{project_memory_content}'，知识类型='project_memory'，来源='团队会议'，标签=['技术栈', '决策', 'React', 'Node.js']",
            deps=deps
        )
        print(f"✅ 项目记忆添加结果: {result.output}")
        
        # 测试2: 添加外部研究资料
        print("\n📚 测试2: 添加外部研究资料...")
        research_content = "根据2024年教育技术报告，个性化学习平台的市场规模预计将达到500亿美元，年增长率为15%。"
        
        result = await knowledge_agent.agent.run(
            f"使用add_knowledge工具添加外部研究：标题='教育技术市场报告'，内容='{research_content}'，知识类型='external_research'，来源='行业报告'，标签=['市场分析', '教育技术', '个性化学习']",
            deps=deps
        )
        print(f"✅ 外部研究添加结果: {result.output}")
        
        # 测试3: 获取知识库统计
        print("\n📊 测试3: 获取知识库统计...")
        stats_result = await knowledge_agent.agent.run(
            "使用get_knowledge_stats工具获取知识库统计信息",
            deps=deps
        )
        print(f"✅ 统计信息: {stats_result.output}")
        
        # 测试4: 使用文本搜索（不使用向量搜索）
        print("\n🔍 测试4: 文本搜索...")
        search_result = await knowledge_agent.agent.run(
            "使用search_knowledge工具搜索包含'技术'关键词的知识，设置use_reranker=false",
            deps=deps
        )
        print(f"✅ 搜索结果: {search_result.output}")
        
        print("\n✅ 基本操作测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

async def test_jina_services_only():
    """只测试Jina服务（不涉及MongoDB）"""
    
    jina_api_key = config.embedding.api_key
    if not jina_api_key:
        print("❌ 缺少JINA_API_KEY配置")
        return
    
    print("\n🧪 测试Jina服务...")
    
    from knowledge_agent import JinaEmbeddingService, JinaRerankerService, JinaSegmenterService
    
    # 测试嵌入服务
    print("\n🔤 测试嵌入服务...")
    try:
        embedding_service = JinaEmbeddingService(
            config.embedding.api_key, 
            config.embedding.model
        )
        
        test_texts = [
            "人工智能在教育中的应用",
            "机器学习算法优化",
            "深度学习模型训练"
        ]
        embeddings = await embedding_service.get_embeddings(test_texts)
        print(f"✅ 生成了 {len(embeddings)} 个嵌入向量")
        print(f"   第一个嵌入向量维度: {len(embeddings[0])}")
        print(f"   第一个嵌入向量前5个值: {embeddings[0][:5]}")
    except Exception as e:
        print(f"❌ 嵌入服务测试失败: {e}")
    
    # 测试重排序服务
    print("\n🔀 测试重排序服务...")
    try:
        reranker_service = JinaRerankerService(config.embedding.api_key)
        
        query = "如何提高教学效果"
        documents = [
            "个性化学习系统可以根据学生的学习进度调整教学内容",
            "人工智能技术可以自动评估学生的学习成果",
            "虚拟现实技术在教育中创造沉浸式学习体验",
            "大数据分析帮助教师了解学生的学习模式"
        ]
        
        reranked_results = await reranker_service.rerank(query, documents, top_n=3)
        
        print(f"✅ 查询: {query}")
        print(f"   重排序后前3个结果:")
        for i, result in enumerate(reranked_results):
            print(f"     {i+1}. 分数: {result['relevance_score']:.4f}")
            print(f"        文档: {result['document']['text']}")
    except Exception as e:
        print(f"❌ 重排序服务测试失败: {e}")
    
    # 测试分割服务
    print("\n✂️ 测试文本分割服务...")
    try:
        segmenter_service = JinaSegmenterService(config.embedding.api_key)
        
        long_text = """
        大学生创新创业教育是培养创新型人才的重要途径。通过项目化学习，学生可以在实践中培养创新思维和创业能力。
        创新创业项目通常包括创新训练项目、创业训练项目和创业实践项目三个层次。
        创新训练项目重点培养学生的科研能力和创新思维，要求学生在导师指导下完成具有创新性的研究工作。
        创业训练项目注重商业模式设计和市场调研，帮助学生了解创业的基本流程和要素。
        创业实践项目是最高层次的实践活动，要求学生实际注册公司并进行商业运营。
        成功的创新创业项目需要具备明确的创新点、可行的技术方案、清晰的商业模式和完整的团队配置。
        项目团队应该包括技术开发、市场营销、财务管理等不同专业背景的成员。
        在项目实施过程中，需要注重知识产权保护、风险管控和可持续发展等关键要素。
        """
        
        chunks = await segmenter_service.segment_text(long_text, max_chunk_length=300)
        print(f"✅ 将长文本分割成 {len(chunks)} 个片段")
        for i, chunk in enumerate(chunks):
            print(f"     片段 {i+1} (长度: {len(chunk)}): {chunk[:100]}...")
    except Exception as e:
        print(f"❌ 分割服务测试失败: {e}")

async def test_knowledge_workflow():
    """测试知识管理工作流"""
    print("\n🔄 测试知识管理工作流...")
    
    if not config.embedding.api_key:
        print("❌ 缺少JINA_API_KEY配置，跳过工作流测试")
        return
    
    knowledge_agent = KnowledgeAgent()
    
    # 模拟外部搜索数据
    external_data = [
        {
            "title": "AI教育应用趋势报告",
            "content": "人工智能在教育领域的应用正在快速发展，包括智能tutoring系统、自适应学习平台和智能评估工具。",
            "source": "web_search",
            "url": "https://example.com/ai-education-trends"
        },
        {
            "title": "个性化学习技术研究",
            "content": "基于学习者模型的个性化推荐算法能够显著提高学习效果，特别是在STEM学科中。",
            "source": "arxiv",
            "url": "https://arxiv.org/abs/example"
        }
    ]
    
    try:
        # 测试处理外部数据
        print("\n📥 测试处理外部数据...")
        result = await knowledge_agent.process_external_data(external_data)
        print(f"✅ 外部数据处理结果: {result[:200]}...")
        
        # 测试项目记忆管理
        print("\n🧠 测试项目记忆管理...")
        memory_result = await knowledge_agent.manage_project_memory(
            "add",
            "项目技术栈确定：使用Python + React + MongoDB开发AI教育平台，预计开发周期6个月"
        )
        print(f"✅ 项目记忆管理结果: {memory_result[:200]}...")
        
    except Exception as e:
        print(f"❌ 工作流测试失败: {e}")

def test_knowledge_dependencies():
    """测试知识库依赖配置"""
    print("\n⚙️ 测试知识库依赖配置...")
    
    # 测试默认配置
    deps = KnowledgeDependencies(
        mongodb_uri=config.database.mongodb_uri,
        database_name="test_db",
        jina_api_key=config.embedding.api_key
    )
    
    print(f"✅ 知识库依赖配置:")
    print(f"   数据库URI: {deps.mongodb_uri}")
    print(f"   数据库名称: {deps.database_name}")
    print(f"   嵌入模型: {deps.embedding_model}")
    print(f"   嵌入维度: {deps.embedding_dimensions}")
    print(f"   最大结果数: {deps.max_results}")
    print(f"   API密钥已配置: {'是' if deps.jina_api_key else '否'}")

async def main():
    """主测试函数"""
    print_knowledge_test_banner()
    
    # 检查配置
    if not validate_config():
        print("⚠️ 配置验证失败，某些测试可能无法运行")
    
    # 运行测试
    test_knowledge_dependencies()
    
    if config.embedding.api_key:
        await test_jina_services_only()
        await test_basic_knowledge_operations()
        await test_knowledge_workflow()
    else:
        print("\n⚠️ 缺少Jina API密钥，跳过知识库在线测试")
        print("请在 .env 文件中配置 JINA_API_KEY")
    
    print("\n🎉 知识库代理测试完成！")
    print("\n📖 知识库功能特点:")
    print("   • 📚 双重知识库：项目记忆 + 外部资料")
    print("   • 🧠 智能嵌入：文本向量化存储")
    print("   • 🔍 灵活搜索：向量搜索 + 文本搜索")
    print("   • 🔀 智能重排：结果相关性优化")
    print("   • ✂️ 文本分割：长文档智能处理")
    print("   • 📊 统计分析：知识库使用情况")

if __name__ == "__main__":
    asyncio.run(main())
