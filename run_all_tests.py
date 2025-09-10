"""
CreatPartner 完整测试套件运行器
基于现有项目架构，统一测试配置系统和多代理功能
"""

import asyncio
import subprocess
import sys
from config import config, validate_config

def print_test_banner():
    """打印测试横幅"""
    print("🧪 CreatPartner 测试套件")
    print("=" * 60)
    print("📋 测试项目：")
    print("   • 🔧 配置系统测试")
    print("   • 🔍 搜索代理测试")
    print("   • 📚 知识库代理测试（简化版）")
    print("   • 🤖 多代理协作测试")
    print("=" * 60)

def check_environment():
    """检查环境配置"""
    print("\n🔧 检查环境配置...")
    
    # 基本配置检查
    config_status = validate_config()
    print(f"✅ 配置系统: {'正常' if config_status else '⚠️ 有警告'}")
    
    # API密钥检查
    api_status = []
    if config.llm.api_key:
        api_status.append("✅ LLM API密钥")
    else:
        api_status.append("❌ LLM API密钥")
    
    if config.embedding.api_key:
        api_status.append("✅ Jina API密钥")
    else:
        api_status.append("❌ Jina API密钥")
    
    print("🔑 API密钥状态:")
    for status in api_status:
        print(f"   {status}")
    
    # 数据库连接检查
    print(f"🗃️ 数据库URI: {config.database.mongodb_uri}")
    
    return config_status

def run_test_script(script_name, description):
    """运行单个测试脚本"""
    print(f"\n{'='*20} {description} {'='*20}")
    try:
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout:
                print("📄 输出:")
                print(result.stdout)
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print("❌ 错误:")
                print(result.stderr)
            if result.stdout:
                print("📄 输出:")
                print(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - 超时")
    except Exception as e:
        print(f"💥 {description} - 异常: {e}")

def main():
    """主测试函数"""
    print_test_banner()
    
    # 检查环境
    env_ok = check_environment()
    if not env_ok:
        print("\n⚠️ 环境配置有问题，某些测试可能失败")
        print("请检查 .env 文件中的配置")
    
    # 测试脚本列表
    test_scripts = [
        ("test_config.py", "配置系统测试"),
        ("test_search_agent.py", "搜索代理测试"),
        ("test_simple_knowledge.py", "知识库代理测试"),
        ("test_multi_agent.py", "多代理协作测试")
    ]
    
    print(f"\n🚀 开始运行 {len(test_scripts)} 个测试脚本...")
    
    # 运行所有测试
    success_count = 0
    for script, description in test_scripts:
        try:
            run_test_script(script, description)
            success_count += 1
        except KeyboardInterrupt:
            print("\n⏹️ 测试被用户中断")
            break
    
    # 测试总结
    print(f"\n{'='*60}")
    print(f"🎯 测试完成总结:")
    print(f"   📊 总测试数: {len(test_scripts)}")
    print(f"   ✅ 运行成功: {success_count}")
    print(f"   ❌ 运行失败: {len(test_scripts) - success_count}")
    
    # 系统建议
    print(f"\n💡 系统功能特点:")
    print(f"   🔧 配置管理: 集中化配置，支持自定义LLM API")
    print(f"   🔍 智能搜索: 网络搜索 + ArXiv学术搜索")
    print(f"   📚 知识管理: 项目记忆 + 外部资料双库")
    print(f"   🤖 多代理协作: 主代理 + 搜索代理 + 知识代理")
    print(f"   🌐 服务集成: SiliconFlow LLM + Jina AI服务")
    
    if not config.llm.api_key:
        print(f"\n📝 配置提示:")
        print(f"   请在 .env 文件中配置以下环境变量:")
        print(f"   • SILICONFLOW_API_KEY=你的API密钥")
        print(f"   • JINA_API_KEY=你的Jina密钥")
        print(f"   • MONGODB_URI=你的MongoDB连接")
    
    print(f"\n🎉 CreatPartner 测试套件运行完成！")

if __name__ == "__main__":
    main()
