"""
测试配置文件的功能
基于现有项目架构重新构建
"""

from config import config, validate_config, print_config, get_model_name, create_llm_provider

def test_config():
    """测试配置功能"""
    print("=" * 50)
    print("CreatPartner 配置测试")
    print("=" * 50)
    
    # 打印当前配置
    print_config()
    
    # 测试模型名称生成
    print(f"\n模型名称: {get_model_name()}")
    
    # 测试配置验证
    print(f"配置验证: {'✅ 通过' if validate_config() else '❌ 失败'}")
    
    # 测试LLM提供商创建
    try:
        provider = create_llm_provider()
        print(f"LLM提供商: {'✅ 创建成功' if provider else '❌ 创建失败'}")
    except Exception as e:
        print(f"LLM提供商: ❌ 创建失败 - {e}")
    
    print("\n配置项详情:")
    print(f"  LLM提供商: {config.llm.provider}")
    print(f"  模型名称: {config.llm.model_name}")
    print(f"  Base URL: {config.llm.base_url}")
    print(f"  温度: {config.llm.temperature}")
    print(f"  最大重试: {config.llm.max_retries}")
    print(f"  超时: {config.llm.timeout}s")
    
    print(f"\n数据库配置:")
    print(f"  MongoDB URI: {config.database.mongodb_uri}")
    print(f"  数据库名称: {config.database.database_name}")
    
    print(f"\n嵌入服务配置:")
    print(f"  提供商: {config.embedding.provider}")
    print(f"  模型: {config.embedding.model}")
    print(f"  维度: {config.embedding.dimensions}")
    
    print(f"\n搜索配置:")
    print(f"  最大结果数: {config.search.max_results}")
    print(f"  启用深度搜索: {config.search.enable_deep_search}")
    print(f"  启用内容提取: {config.search.enable_content_extraction}")
    print(f"  启用重排序: {config.search.enable_reranker}")
    
    print(f"\n项目配置:")
    print(f"  默认项目名称: {config.project.default_project_name}")
    print(f"  默认项目阶段: {config.project.default_project_stage}")
    print(f"  每会话最大请求数: {config.project.max_requests_per_session}")
    print(f"  每会话最大Token数: {config.project.max_tokens_per_session}")
    print(f"  每会话最大工具调用数: {config.project.max_tool_calls_per_session}")


def test_environment_variables():
    """测试环境变量"""
    print("\n" + "=" * 50)
    print("环境变量检查")
    print("=" * 50)
    
    import os
    
    required_vars = [
        "SILICONFLOW_API_KEY",
        "JINA_API_KEY", 
        "MONGODB_URI"
    ]
    
    optional_vars = [
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "DB_NAME",
        "MAX_SEARCH_RESULTS",
        "ENABLE_DEEP_SEARCH",
        "DEBUG"
    ]
    
    print("必需的环境变量:")
    for var in required_vars:
        value = os.getenv(var)
        status = "✅ 已设置" if value else "❌ 未设置"
        print(f"  {var}: {status}")
    
    print("\n可选的环境变量:")
    for var in optional_vars:
        value = os.getenv(var)
        status = f"✅ {value}" if value else "⚪ 未设置"
        print(f"  {var}: {status}")


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 50)
    print("模型创建测试")
    print("=" * 50)
    
    try:
        print("创建CreatPartner代理...")
        from main_agent import create_creatpartner_agent
        agent = create_creatpartner_agent()
        print("✅ CreatPartner代理创建成功")
        
        print("创建搜索代理...")
        from search_agent import create_search_agent
        search_agent = create_search_agent()
        print("✅ 搜索代理创建成功")
        
        print("创建知识库代理...")
        from knowledge_agent import create_knowledge_agent
        knowledge_agent = create_knowledge_agent()
        print("✅ 知识库代理创建成功")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()


def test_config_integration():
    """测试配置与项目代码的集成"""
    print("\n" + "=" * 50)
    print("配置集成测试")
    print("=" * 50)
    
    try:
        # 测试共享依赖创建
        print("测试共享依赖创建...")
        from main_agent import create_shared_dependencies
        deps = create_shared_dependencies(
            project_name="测试项目",
            project_description="用于测试配置集成",
            project_stage="planning"
        )
        print("✅ 共享依赖创建成功")
        print(f"  项目名称: {deps.project_name}")
        print(f"  API密钥已配置: {'是' if deps.jina_api_key else '否'}")
        print(f"  数据库URI: {deps.mongodb_uri}")
        
        # 测试搜索依赖创建
        print("\n测试搜索依赖创建...")
        from search_agent import create_search_dependencies
        search_deps = create_search_dependencies(
            max_results=3,
            enable_deep_search=False
        )
        print("✅ 搜索依赖创建成功")
        print(f"  最大结果数: {search_deps.max_results}")
        print(f"  API密钥已配置: {'是' if search_deps.jina_api_key else '否'}")
        
    except Exception as e:
        print(f"❌ 配置集成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_config()
    test_environment_variables()
    test_config_integration()
    test_model_creation()
