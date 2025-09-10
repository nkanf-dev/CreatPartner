"""
搜索代理 - 基于pydantic-ai的智能搜索助手
集成Jina AI完整生态：Search、Reader、DeepSearch、Classifier等
支持web搜索、网页内容提取、学术论文搜索和智能内容分析
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal

# 导入配置
from config import config, get_model_name, create_llm_provider

try:
    import arxiv
    import httpx
    from pydantic import BaseModel
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIChatModel
    
except ImportError as e:
    print(f"警告: 缺少依赖包 {e}. 请运行: uv add pydantic-ai httpx arxiv python-dotenv")
    # 创建模拟类以避免导入错误
    class BaseModel:
        pass
    class Agent:
        def __init__(self, *args, **kwargs):
            pass
        def tool(self, func):
            return func
        def run_sync(self, *args, **kwargs):
            return type('Result', (), {'output': '模拟结果 - 请安装依赖包'})()
    class RunContext:
        pass


class SearchResult(BaseModel):
    """搜索结果的基础模型"""
    title: str = ""
    url: str = ""
    content: str = ""
    source: str = ""  # "web", "arxiv", "reader", "deepsearch"
    

class WebSearchResult(SearchResult):
    """Web搜索结果"""
    snippet: str = ""
    description: str = ""
    

class ReaderResult(SearchResult):
    """网页内容提取结果"""
    images: Dict[str, str] = {}
    links: Dict[str, str] = {}
    

class ArxivSearchResult(SearchResult):
    """Arxiv搜索结果"""
    authors: List[str] = []
    published: str = ""
    summary: str = ""


class DeepSearchResult(SearchResult):
    """DeepSearch结果"""
    reasoning_steps: List[str] = []
    urls_used: List[str] = []
    

@dataclass
class SearchDependencies:
    """搜索代理的依赖"""
    jina_api_key: Optional[str] = None
    max_results: int = 5
    enable_deep_search: bool = False
    enable_content_extraction: bool = True
    

class SearchAgent:
    """搜索代理类 - 基于Jina AI完整生态的生产级实现"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_model_name()
        
        # 创建自定义模型实例
        model = self._create_model(model_name)
            
        self.agent = Agent(
            model,
            deps_type=SearchDependencies,
            system_prompt="""
            你是一个专业的学术研究助手，专门为大学生创新创业竞赛提供信息检索服务。
            
            你拥有强大的信息检索能力：
            1. Jina Search API - 进行高质量的网络搜索
            2. Jina Reader API - 提取和解析网页内容
            3. Jina DeepSearch API - 进行深度研究和推理
            4. Jina Classifier API - 对内容进行智能分类
            5. Arxiv API - 获取权威学术论文
            
            你的工作原则：
            - 根据查询类型选择最合适的搜索策略
            - 优先获取权威、可信的信息源
            - 提供结构化和易于理解的搜索结果
            - 特别关注创新性、技术可行性和商业价值
            - 确保信息的时效性和准确性
            
            搜索策略：
            - 学术研究：优先使用Arxiv + Jina Search
            - 市场调研：使用Jina Search + Reader提取详细内容
            - 技术分析：使用DeepSearch进行深度分析
            - 内容分类：使用Classifier API进行智能归类
            """,
        )
        
        # 注册搜索工具
        self._register_tools()
    
    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                # 使用自定义提供商
                provider = create_llm_provider()
                if provider:
                    return OpenAIChatModel(
                        config.llm.model_name,
                        provider=provider
                    )
            
            # 回退到默认模型
            return model_name
        except Exception as e:
            print(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name
    
    def _register_tools(self):
        """注册搜索工具"""
        
        @self.agent.tool
        async def jina_search(
            ctx: RunContext[SearchDependencies], 
            query: str,
            max_results: Optional[int] = None,
            country_code: str = "US",
            language: str = "en"
        ) -> List[Dict[str, Any]]:
            """使用Jina Search API进行web搜索
            
            Args:
                query: 搜索查询
                max_results: 最大结果数量
                country_code: 国家代码(如US, CN)
                language: 语言代码(如en, zh)
                
            Returns:
                搜索结果列表
            """
            max_results = max_results or ctx.deps.max_results
            
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"
                
                payload = {
                    "q": query,
                    "gl": country_code,
                    "hl": language,
                    "num": max_results
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://s.jina.ai/",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = []
                        
                        for item in data.get("data", []):
                            results.append({
                                "title": item.get("title", ""),
                                "content": item.get("content", "")[:1200],
                                "url": item.get("url", ""),
                                "description": item.get("description", ""),
                                "source": "jina_search",
                                "snippet": item.get("content", "")[:300]
                            })
                        
                        return results
                    else:
                        return [{"title": "搜索失败", "content": f"HTTP {response.status_code}: {response.text}", "source": "jina_search"}]
                        
            except Exception as e:
                return [{"title": "Jina搜索错误", "content": f"错误详情: {str(e)}", "source": "jina_search"}]
        
        @self.agent.tool
        async def jina_reader(
            ctx: RunContext[SearchDependencies],
            url: str,
            return_format: Literal["markdown", "html", "text"] = "markdown",
            include_images: bool = True,
            include_links: bool = True
        ) -> Dict[str, Any]:
            """使用Jina Reader API提取网页内容
            
            Args:
                url: 要提取的网页URL
                return_format: 返回格式
                include_images: 是否包含图片信息
                include_links: 是否包含链接信息
                
            Returns:
                网页内容提取结果
            """
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Return-Format": return_format
                }
                
                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"
                
                if include_images:
                    headers["X-With-Images-Summary"] = "true"
                if include_links:
                    headers["X-With-Links-Summary"] = "true"
                
                payload = {"url": url}
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://r.jina.ai/",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        page_data = data.get("data", {})
                        
                        return {
                            "title": page_data.get("title", ""),
                            "content": page_data.get("content", ""),
                            "url": page_data.get("url", url),
                            "description": page_data.get("description", ""),
                            "images": page_data.get("images", {}),
                            "links": page_data.get("links", {}),
                            "source": "jina_reader"
                        }
                    else:
                        return {
                            "title": "内容提取失败",
                            "content": f"HTTP {response.status_code}: {response.text}",
                            "url": url,
                            "source": "jina_reader"
                        }
                        
            except Exception as e:
                return {
                    "title": "Reader错误",
                    "content": f"错误详情: {str(e)}",
                    "url": url,
                    "source": "jina_reader"
                }
        
        @self.agent.tool
        async def jina_deepsearch(
            ctx: RunContext[SearchDependencies],
            query: str,
            reasoning_effort: Literal["low", "medium", "high"] = "medium",
            max_urls: int = 5
        ) -> Dict[str, Any]:
            """使用Jina DeepSearch API进行深度研究
            
            Args:
                query: 研究查询
                reasoning_effort: 推理努力程度
                max_urls: 最大返回URL数量
                
            Returns:
                深度研究结果
            """
            if not ctx.deps.enable_deep_search:
                return {
                    "title": "DeepSearch未启用",
                    "content": "请在依赖配置中启用enable_deep_search",
                    "source": "jina_deepsearch"
                }
            
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"
                
                payload = {
                    "model": "jina-deepsearch-v1",
                    "messages": [
                        {"role": "user", "content": query}
                    ],
                    "reasoning_effort": reasoning_effort,
                    "max_returned_urls": max_urls,
                    "stream": False
                }
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        "https://deepsearch.jina.ai/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            
                            return {
                                "title": f"DeepSearch: {query[:50]}...",
                                "content": content,
                                "source": "jina_deepsearch",
                                "reasoning_steps": [],  # DeepSearch会在content中包含推理过程
                                "urls_used": []  # 可以从content中提取URL
                            }
                        else:
                            return {
                                "title": "DeepSearch无结果",
                                "content": "未获得有效的搜索结果",
                                "source": "jina_deepsearch"
                            }
                    else:
                        return {
                            "title": "DeepSearch失败",
                            "content": f"HTTP {response.status_code}: {response.text}",
                            "source": "jina_deepsearch"
                        }
                        
            except Exception as e:
                return {
                    "title": "DeepSearch错误",
                    "content": f"错误详情: {str(e)}",
                    "source": "jina_deepsearch"
                }
        
        @self.agent.tool
        async def jina_classify(
            ctx: RunContext[SearchDependencies],
            texts: List[str],
            labels: List[str],
            model: str = "jina-embeddings-v3"
        ) -> List[Dict[str, Any]]:
            """使用Jina Classifier API对文本进行分类
            
            Args:
                texts: 要分类的文本列表
                labels: 分类标签列表
                model: 使用的模型
                
            Returns:
                分类结果列表
            """
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"
                
                payload = {
                    "model": model,
                    "input": texts,
                    "labels": labels
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.jina.ai/v1/classify",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = []
                        
                        for item in data.get("data", []):
                            results.append({
                                "index": item.get("index", 0),
                                "prediction": item.get("prediction", ""),
                                "score": item.get("score", 0.0),
                                "predictions": item.get("predictions", [])
                            })
                        
                        return results
                    else:
                        return [{"error": f"HTTP {response.status_code}: {response.text}"}]
                        
            except Exception as e:
                return [{"error": f"分类错误: {str(e)}"}]
        
        @self.agent.tool
        async def arxiv_search(
            ctx: RunContext[SearchDependencies],
            query: str,
            max_results: Optional[int] = None,
            sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance"
        ) -> List[Dict[str, Any]]:
            """使用arxiv API搜索学术论文
            
            Args:
                query: 搜索查询
                max_results: 最大结果数量
                sort_by: 排序方式
                
            Returns:
                论文搜索结果列表
            """
            max_results = max_results or ctx.deps.max_results
            
            try:
                # 设置排序标准
                sort_criterion = arxiv.SortCriterion.Relevance
                if sort_by == "lastUpdatedDate":
                    sort_criterion = arxiv.SortCriterion.LastUpdatedDate
                elif sort_by == "submittedDate":
                    sort_criterion = arxiv.SortCriterion.SubmittedDate
                
                # 使用arxiv库进行搜索
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=sort_criterion
                )
                
                results = []
                for paper in search.results():
                    results.append({
                        "title": paper.title.strip(),
                        "content": paper.summary[:1200],
                        "source": "arxiv",
                        "url": paper.entry_id,
                        "authors": [author.name for author in paper.authors],
                        "published": paper.published.strftime("%Y-%m-%d"),
                        "summary": paper.summary.strip(),
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url
                    })
                
                return results
                
            except Exception as e:
                return [{"title": "Arxiv搜索错误", "content": f"错误详情: {str(e)}", "source": "arxiv"}]
        
        @self.agent.tool
        async def comprehensive_search(
            ctx: RunContext[SearchDependencies],
            query: str,
            search_types: List[Literal["jina_search", "arxiv", "deepsearch"]] = ["jina_search", "arxiv"],
            max_results_per_type: int = 3
        ) -> Dict[str, Any]:
            """综合搜索 - 整合多种搜索源
            
            Args:
                query: 搜索查询
                search_types: 要使用的搜索类型
                max_results_per_type: 每种类型的最大结果数
                
            Returns:
                综合搜索结果
            """
            results = {"query": query, "sources": {}}
            
            try:
                # Jina搜索
                if "jina_search" in search_types:
                    jina_results = await jina_search(ctx, query, max_results_per_type)
                    results["sources"]["jina_search"] = jina_results
                
                # Arxiv搜索
                if "arxiv" in search_types:
                    arxiv_results = await arxiv_search(ctx, query, max_results_per_type)
                    results["sources"]["arxiv"] = arxiv_results
                
                # DeepSearch
                if "deepsearch" in search_types and ctx.deps.enable_deep_search:
                    deepsearch_result = await jina_deepsearch(ctx, query)
                    results["sources"]["deepsearch"] = [deepsearch_result]
                
                # 统计总结果数
                total_results = sum(len(source_results) for source_results in results["sources"].values())
                results["total_results"] = total_results
                results["search_types_used"] = search_types
                
                return results
                
            except Exception as e:
                return {
                    "query": query,
                    "error": f"综合搜索错误: {str(e)}",
                    "sources": {}
                }
        
        @self.agent.tool
        async def extract_and_analyze_urls(
            ctx: RunContext[SearchDependencies],
            urls: List[str],
            analysis_labels: List[str] = ["技术创新", "商业价值", "可行性分析", "风险评估"]
        ) -> List[Dict[str, Any]]:
            """提取URL内容并进行智能分析
            
            Args:
                urls: 要分析的URL列表
                analysis_labels: 分析标签
                
            Returns:
                URL内容提取和分析结果
            """
            results = []
            
            for url in urls[:5]:  # 限制最多5个URL以控制成本
                try:
                    # 使用Reader API提取内容
                    content_result = await jina_reader(ctx, url, "text", False, True)
                    
                    if content_result.get("content"):
                        # 对内容进行分类分析
                        content_text = content_result["content"][:2000]  # 限制长度
                        classification_results = await jina_classify(
                            ctx, [content_text], analysis_labels
                        )
                        
                        result = {
                            "url": url,
                            "title": content_result.get("title", ""),
                            "content_summary": content_text[:500],
                            "analysis": classification_results[0] if classification_results else {},
                            "links": content_result.get("links", {}),
                            "extraction_success": True
                        }
                    else:
                        result = {
                            "url": url,
                            "title": "提取失败",
                            "content_summary": "",
                            "analysis": {},
                            "links": {},
                            "extraction_success": False,
                            "error": content_result.get("content", "未知错误")
                        }
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        "url": url,
                        "title": "处理错误",
                        "content_summary": "",
                        "analysis": {},
                        "links": {},
                        "extraction_success": False,
                        "error": str(e)
                    })
            
            return results
    
    async def search(
        self, 
        query: str, 
        deps: Optional[SearchDependencies] = None,
        search_type: Literal["comprehensive", "jina_search", "arxiv", "deepsearch"] = "comprehensive"
    ) -> str:
        """异步执行搜索任务"""
        if deps is None:
            deps = SearchDependencies(
                jina_api_key=config.embedding.api_key,
                max_results=config.search.max_results,
                enable_deep_search=config.search.enable_deep_search,
                enable_content_extraction=config.search.enable_content_extraction
            )
        
        # 根据搜索类型调整查询
        if search_type == "comprehensive":
            search_query = f"请使用comprehensive_search工具搜索: {query}"
        elif search_type == "jina_search":
            search_query = f"请使用jina_search工具搜索: {query}"
        elif search_type == "arxiv":
            search_query = f"请使用arxiv_search工具搜索学术论文: {query}"
        elif search_type == "deepsearch":
            search_query = f"请使用jina_deepsearch工具进行深度研究: {query}"
        else:
            search_query = query
            
        result = await self.agent.run(search_query, deps=deps)
        return result.output
    
    def search_sync(
        self, 
        query: str, 
        deps: Optional[SearchDependencies] = None,
        search_type: Literal["comprehensive", "jina_search", "arxiv", "deepsearch"] = "comprehensive"
    ) -> str:
        """同步执行搜索任务"""
        if deps is None:
            deps = SearchDependencies(
                jina_api_key=config.embedding.api_key,
                max_results=config.search.max_results,
                enable_deep_search=config.search.enable_deep_search,
                enable_content_extraction=config.search.enable_content_extraction
            )
        
        # 根据搜索类型调整查询
        if search_type == "comprehensive":
            search_query = f"请使用comprehensive_search工具搜索: {query}"
        elif search_type == "jina_search":
            search_query = f"请使用jina_search工具搜索: {query}"
        elif search_type == "arxiv":
            search_query = f"请使用arxiv_search工具搜索学术论文: {query}"
        elif search_type == "deepsearch":
            search_query = f"请使用jina_deepsearch工具进行深度研究: {query}"
        else:
            search_query = query
            
        result = self.agent.run_sync(search_query, deps=deps)
        return result.output
    
    async def extract_content(
        self,
        urls: List[str],
        deps: Optional[SearchDependencies] = None,
        analyze: bool = True
    ) -> List[Dict[str, Any]]:
        """提取URLs内容的便利方法"""
        if deps is None:
            deps = SearchDependencies(
                jina_api_key=config.embedding.api_key,
                enable_content_extraction=config.search.enable_content_extraction
            )
        
        if analyze:
            result = await self.agent.run(
                f"请使用extract_and_analyze_urls工具分析这些URL: {urls}",
                deps=deps
            )
        else:
            # 简单提取内容
            results = []
            for url in urls:
                content_result = await self.agent.run(
                    f"请使用jina_reader工具提取URL内容: {url}",
                    deps=deps
                )
                results.append(content_result.output)
            result = {"results": results}
        
        return result if isinstance(result, dict) else {"output": result}


# 工厂函数
def create_search_agent(model_name: str = None) -> SearchAgent:
    """创建搜索代理实例"""
    return SearchAgent(model_name)


# 便利函数：创建标准搜索依赖
def create_search_dependencies(
    jina_api_key: str = None,
    max_results: int = 5,
    enable_deep_search: bool = False,
    enable_content_extraction: bool = True
) -> SearchDependencies:
    """创建搜索依赖配置"""
    return SearchDependencies(
        jina_api_key=jina_api_key or config.embedding.api_key,
        max_results=max_results,
        enable_deep_search=enable_deep_search,
        enable_content_extraction=enable_content_extraction
    )


# 使用示例
async def main():
    """搜索代理使用示例"""
    print("🔍 CreatPartner 搜索代理演示")
    print("=" * 50)
    
    # 创建搜索代理
    agent = create_search_agent()
    
    # 创建依赖配置
    deps = create_search_dependencies(
        max_results=3,
        enable_deep_search=True  # 启用深度搜索
    )
    
    # 演示不同类型的搜索
    queries = [
        ("AI在教育中的应用", "comprehensive"),
        ("machine learning in education", "arxiv"),
        ("人工智能教育趋势分析", "deepsearch")
    ]
    
    for query, search_type in queries:
        print(f"\n📝 {search_type}搜索: {query}")
        try:
            result = await agent.search(query, deps, search_type)
            print(f"✅ 搜索完成")
            print(f"📊 结果摘要: {result[:200]}...")
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
    
    # 演示内容提取
    test_urls = [
        "https://jina.ai",
        "https://arxiv.org/abs/2301.00000"  # 示例URL
    ]
    
    print(f"\n🌐 内容提取演示: {len(test_urls)} 个URL")
    try:
        extraction_results = await agent.extract_content(test_urls, deps, analyze=True)
        print("✅ 内容提取完成")
        print(f"📋 提取结果数量: {len(extraction_results.get('results', []))}")
    except Exception as e:
        print(f"❌ 内容提取失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
