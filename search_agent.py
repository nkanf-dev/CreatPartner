import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal, AsyncIterator
from config import config, get_model_name, create_llm_provider
from logger import (
    get_logger,
    debug,
    info,
    warning,
    error,
    success,
    search_operation,
    agent_operation,
)

try:
    import arxiv
    import httpx
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext, ModelRetry
    from pydantic_ai.models.openai import OpenAIChatModel

except ImportError as e:
    warning(f"缺少依赖包 {e}. 请运行: uv add pydantic-ai httpx arxiv python-dotenv")

    # 创建模拟类以避免导入错误
    class BaseModel:
        pass

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, func):
            return func

        def instructions(self, func):
            return func

        def run_sync(self, *args, **kwargs):
            return type("Result", (), {"output": "模拟结果 - 请安装依赖包"})()

    class RunContext:
        pass

    class ModelRetry:
        pass


class SearchResult(BaseModel):
    """搜索结果的基础模型"""

    title: str = ""
    url: str = ""
    content: str = ""
    source: str = ""  # "web", "arxiv", "reader"


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


@dataclass
class SearchDependencies:
    """搜索代理的依赖"""

    jina_api_key: Optional[str] = None
    max_results: int = 5
    enable_content_extraction: bool = True


class SearchAgent:
    """搜索代理类 - 基于Jina AI完整生态的生产级实现"""

    def __init__(self, model_name: str = None):
        if config.project.debug_mode:
            agent_operation("搜索代理", "初始化", "开始")

        if model_name is None:
            model_name = get_model_name()

        # 创建自定义模型实例
        model = self._create_model(model_name)

        self.agent = Agent(
            model,
            deps_type=SearchDependencies,
            instructions="""
            你是一个专业的学术研究助手，专门为大学生创新创业竞赛提供信息检索服务。
            
            你拥有强大的信息检索能力：
            1. Jina Search API - 进行高质量的网络搜索
            2. Jina Reader API - 提取和解析网页内容
            3. Jina Classifier API - 对内容进行智能分类
            4. Arxiv API - 获取权威学术论文
            
            重要原则：
            - 严格按照用户请求执行单次搜索，不要自动扩展或重复搜索
            - 一次工具调用完成任务，除非明确要求多轮搜索
            - 根据查询类型选择最合适的搜索策略
            - 优先获取权威、可信的信息源
            - 提供结构化和易于理解的搜索结果
            - 关注信息的准确性和相关性
            
            搜索策略：
            - 学术研究：优先使用Arxiv + Jina Search（一次comprehensive_search调用）
            - 市场调研：使用Jina Search + Reader提取详细内容（一次comprehensive_search调用）
            - 技术分析：使用Jina Search进行综合分析（一次comprehensive_search调用）
            - 内容分类：使用Classifier API进行智能归类
            
            执行控制：
            - 接收到comprehensive_search请求时，执行一次工具调用即可
            - 不要基于搜索结果自动进行额外的搜索
            - 如果需要更多信息，在响应中说明而不是自动搜索
            """,
            # 设置重试机制
            retries=1,  # 降低重试次数，减少重复
        )

        # 注册搜索工具
        self._register_tools()

        if config.project.debug_mode:
            agent_operation("搜索代理", "初始化", "完成")

    def _create_model(self, model_name: str):
        """创建自定义LLM模型实例"""
        try:
            if config.llm.provider in ["siliconflow", "deepseek"]:
                # 使用自定义提供商
                provider = create_llm_provider()
                if provider:
                    return OpenAIChatModel(config.llm.model_name, provider=provider)

            # 回退到默认模型
            return model_name
        except Exception as e:
            error(f"创建自定义模型失败，使用默认模型: {e}")
            return model_name

    def _register_tools(self):
        """注册搜索工具"""

        @self.agent.tool
        async def jina_search(
            ctx: RunContext[SearchDependencies],
            query: str,
            max_results: Optional[int] = None,
            country_code: str = "US",
            language: str = "en",
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
                if config.project.debug_mode:
                    search_operation(query[:100], max_results, "Jina搜索")

                # 使用GET方法访问s.jina.ai，这是更稳定的方式
                headers = {
                    "Accept": "application/json",
                    "User-Agent": "CreatPartner/1.0",
                }

                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"

                # 构建查询参数
                params = {
                    "q": query,
                    "gl": country_code,
                    "hl": language,
                    "num": min(
                        max_results, config.llm.max_search_results
                    ),  # 使用配置的最大结果数
                }

                # 使用配置的超时时间
                timeout_config = httpx.Timeout(
                    connect=config.llm.timeout * 0.1,  # 连接超时
                    read=config.llm.timeout * 0.3,  # 读取超时
                    write=config.llm.timeout * 0.1,  # 写入超时
                    pool=config.llm.timeout * 0.05,  # 池超时
                )

                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    response = await client.get(
                        "https://s.jina.ai/", headers=headers, params=params
                    )

                    if response.status_code == 200:
                        # 尝试解析JSON响应
                        try:
                            data = response.json()

                            # 处理JSON响应
                            results = []
                            data_items = (
                                data.get("data", []) if isinstance(data, dict) else []
                            )

                            # 如果data是列表
                            if isinstance(data, list):
                                data_items = data

                            for item in data_items:
                                if isinstance(item, dict):
                                    results.append(
                                        {
                                            "title": item.get("title", "")[
                                                : config.llm.max_content_length // 5
                                            ],
                                            "content": item.get("content", "")[
                                                : config.llm.max_content_length
                                            ],
                                            "url": item.get("url", ""),
                                            "description": item.get("description", "")[
                                                : config.llm.max_content_length // 3
                                            ],
                                            "source": "jina_search",
                                            "snippet": item.get("content", "")[
                                                : config.llm.max_content_length // 3
                                            ],
                                        }
                                    )

                            # 如果没有有效结果，但有原始数据，尝试其他解析方式
                            if not results and data:
                                if isinstance(data, dict) and data.get("content"):
                                    results.append(
                                        {
                                            "title": f"搜索结果: {query}",
                                            "content": str(data.get("content", ""))[
                                                :1200
                                            ],
                                            "url": f"https://s.jina.ai/?q={query}",
                                            "description": str(data.get("content", ""))[
                                                :300
                                            ],
                                            "source": "jina_search",
                                            "snippet": str(data.get("content", ""))[
                                                :300
                                            ],
                                        }
                                    )
                                elif isinstance(data, str):
                                    results.append(
                                        {
                                            "title": f"搜索结果: {query}",
                                            "content": data[:1200],
                                            "url": f"https://s.jina.ai/?q={query}",
                                            "description": data[:300],
                                            "source": "jina_search",
                                            "snippet": data[:300],
                                        }
                                    )

                            if config.project.debug_mode:
                                success("Jina搜索完成", results=len(results))

                            return results

                        except Exception as json_error:
                            # 如果不是JSON，可能是文本响应
                            text_content = response.text
                            if config.project.debug_mode:
                                debug(
                                    f"Jina搜索返回文本内容",
                                    length=f"{len(text_content)} 字符",
                                    error=str(json_error),
                                )

                            # 简单解析文本内容
                            results = [
                                {
                                    "title": f"搜索结果: {query}",
                                    "content": text_content[:1200],
                                    "url": f"https://s.jina.ai/?q={query}",
                                    "description": text_content[:300],
                                    "source": "jina_search",
                                    "snippet": text_content[:300],
                                }
                            ]

                            if config.project.debug_mode:
                                success("Jina搜索完成，文本内容已解析")

                            return results
                        else:
                            error_result = [
                                {
                                    "title": "搜索失败",
                                    "content": f"HTTP {response.status_code}: {response.text[:500]}",
                                    "url": "",
                                    "description": f"搜索失败: {response.status_code}",
                                    "source": "jina_search",
                                    "snippet": f"错误: {response.status_code}",
                                }
                            ]
                            if config.project.debug_mode:
                                info("Jina搜索结束", status_code=response.status_code)
                            return error_result

            except Exception as e:
                error_result = [
                    {
                        "title": "Jina搜索错误",
                        "content": f"错误详情: {str(e)}",
                        "url": "",
                        "description": f"搜索异常: {str(e)[:100]}",
                        "source": "jina_search",
                        "snippet": f"异常: {str(e)[:100]}",
                    }
                ]
                if config.project.debug_mode:
                    info("Jina搜索结束", error=str(e))
                return error_result

        @self.agent.tool(retries=1)
        async def jina_reader(
            ctx: RunContext[SearchDependencies],
            url: str,
            return_format: Literal["markdown", "html", "text"] = "markdown",
            include_images: bool = True,
            include_links: bool = True,
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
                if config.project.debug_mode:
                    info("Jina Reader提取", url=url)

                headers = {
                    "Accept": "application/json",
                    "User-Agent": "CreatPartner/1.0",
                }

                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"

                # 构建Jina Reader URL
                reader_url = f"https://r.jina.ai/{url}"

                # 设置超时配置
                timeout_config = httpx.Timeout(
                    connect=10.0, read=30.0, write=10.0, pool=5.0
                )

                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    response = await client.get(reader_url, headers=headers)

                    if response.status_code == 200:
                        content_text = response.text

                        if config.project.debug_mode:
                            success(
                                "Jina Reader提取成功",
                                length=f"{len(content_text)} 字符",
                            )

                        return {
                            "title": f"页面内容: {url}",
                            "content": content_text[:2000],  # 限制内容长度
                            "url": url,
                            "description": content_text[:300],
                            "images": {},  # 简化处理
                            "links": {},  # 简化处理
                            "source": "jina_reader",
                        }
                    else:
                        if config.project.debug_mode:
                            error("Jina Reader失败", status_code=response.status_code)

                        return {
                            "title": "内容提取失败",
                            "content": f"HTTP {response.status_code}: 无法提取页面内容",
                            "url": url,
                            "description": f"提取失败: {response.status_code}",
                            "images": {},
                            "links": {},
                            "source": "jina_reader",
                        }

            except httpx.TimeoutException:
                return {
                    "title": "Reader超时",
                    "content": "页面内容提取超时",
                    "url": url,
                    "description": "提取超时",
                    "images": {},
                    "links": {},
                    "source": "jina_reader",
                }
            except Exception as e:
                if config.project.debug_mode:
                    error("Jina Reader异常", error=str(e))

                return {
                    "title": "Reader错误",
                    "content": f"错误详情: {str(e)}",
                    "url": url,
                    "description": f"提取异常: {str(e)[:100]}",
                    "images": {},
                    "links": {},
                    "source": "jina_reader",
                }

        @self.agent.tool
        async def jina_classify(
            ctx: RunContext[SearchDependencies],
            texts: List[str],
            labels: List[str],
            model: str = "jina-embeddings-v3",
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
                    "Accept": "application/json",
                }

                if ctx.deps.jina_api_key:
                    headers["Authorization"] = f"Bearer {ctx.deps.jina_api_key}"

                payload = {"model": model, "input": texts, "labels": labels}

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.jina.ai/v1/classify", headers=headers, json=payload
                    )

                    if response.status_code == 200:
                        data = response.json()
                        results = []

                        for item in data.get("data", []):
                            results.append(
                                {
                                    "index": item.get("index", 0),
                                    "prediction": item.get("prediction", ""),
                                    "score": item.get("score", 0.0),
                                    "predictions": item.get("predictions", []),
                                }
                            )

                        return results
                    else:
                        return [
                            {"error": f"HTTP {response.status_code}: {response.text}"}
                        ]

            except Exception as e:
                return [{"error": f"分类错误: {str(e)}"}]

        @self.agent.tool
        async def arxiv_search(
            ctx: RunContext[SearchDependencies],
            query: str,
            max_results: Optional[int] = None,
            sort_by: Literal[
                "relevance", "lastUpdatedDate", "submittedDate"
            ] = "relevance",
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
                if config.project.debug_mode:
                    search_operation(query, max_results, "Arxiv搜索")

                # 设置排序标准
                sort_criterion = arxiv.SortCriterion.Relevance
                if sort_by == "lastUpdatedDate":
                    sort_criterion = arxiv.SortCriterion.LastUpdatedDate
                elif sort_by == "submittedDate":
                    sort_criterion = arxiv.SortCriterion.SubmittedDate

                # 使用arxiv库进行搜索
                search = arxiv.Search(
                    query=query, max_results=max_results, sort_by=sort_criterion
                )

                results = []
                try:
                    for paper in search.results():
                        results.append(
                            {
                                "title": paper.title.strip(),
                                "content": paper.summary[:1200],
                                "source": "arxiv",
                                "url": paper.entry_id,
                                "authors": [author.name for author in paper.authors],
                                "published": paper.published.strftime("%Y-%m-%d"),
                                "summary": paper.summary.strip(),
                                "categories": paper.categories,
                                "pdf_url": paper.pdf_url,
                            }
                        )
                except Exception as search_error:
                    if config.project.debug_mode:
                        warning("Arxiv搜索过程中出错", error=str(search_error))

                    # 返回部分结果
                    if results:
                        return results
                    else:
                        return [
                            {
                                "title": "Arxiv搜索部分失败",
                                "content": f"搜索过程中出现问题: {search_error}",
                                "source": "arxiv",
                                "url": "",
                                "error": str(search_error),
                            }
                        ]

                if config.project.debug_mode:
                    success("Arxiv搜索完成", results=len(results))

                return (
                    results
                    if results
                    else [
                        {
                            "title": "未找到相关论文",
                            "content": f"在Arxiv中未找到与'{query}'相关的论文",
                            "source": "arxiv",
                            "url": "",
                        }
                    ]
                )

            except Exception as e:
                error_msg = str(e)
                if config.project.debug_mode:
                    error("Arxiv搜索失败", error=error_msg)

                return [
                    {
                        "title": "Arxiv搜索错误",
                        "content": f"错误详情: {error_msg}",
                        "source": "arxiv",
                        "url": "",
                        "error": error_msg,
                    }
                ]

        @self.agent.tool
        async def comprehensive_search(
            ctx: RunContext[SearchDependencies],
            query: str,
            search_types: List[Literal["jina_search", "arxiv"]] = [
                "jina_search",
                "arxiv",
            ],
            max_results_per_type: int = 3,
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

                # 统计总结果数
                total_results = sum(
                    len(source_results)
                    for source_results in results["sources"].values()
                )
                results["total_results"] = total_results
                results["search_types_used"] = search_types

                return results

            except Exception as e:
                return {
                    "query": query,
                    "error": f"综合搜索错误: {str(e)}",
                    "sources": {},
                }

        @self.agent.tool
        async def extract_and_analyze_urls(
            ctx: RunContext[SearchDependencies],
            urls: List[str],
            analysis_labels: List[str] = [
                "技术创新",
                "商业价值",
                "可行性分析",
                "风险评估",
            ],
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
                            "analysis": classification_results[0]
                            if classification_results
                            else {},
                            "links": content_result.get("links", {}),
                            "extraction_success": True,
                        }
                    else:
                        result = {
                            "url": url,
                            "title": "提取失败",
                            "content_summary": "",
                            "analysis": {},
                            "links": {},
                            "extraction_success": False,
                            "error": content_result.get("content", "未知错误"),
                        }

                    results.append(result)

                except Exception as e:
                    results.append(
                        {
                            "url": url,
                            "title": "处理错误",
                            "content_summary": "",
                            "analysis": {},
                            "links": {},
                            "extraction_success": False,
                            "error": str(e),
                        }
                    )

            return results

    async def search(
        self,
        query: str,
        deps: Optional[SearchDependencies] = None,
        search_type: Literal["comprehensive", "jina_search", "arxiv"] = "comprehensive",
    ) -> str:
        """异步执行搜索任务"""
        if deps is None:
            deps = SearchDependencies(
                jina_api_key=config.embedding.api_key,
                max_results=config.search.max_results,
                enable_content_extraction=config.search.enable_content_extraction,
            )

        # 根据搜索类型调整查询
        if search_type == "comprehensive":
            search_query = f"请使用comprehensive_search工具搜索: {query}"
        elif search_type == "jina_search":
            search_query = f"请使用jina_search工具搜索: {query}"
        elif search_type == "arxiv":
            search_query = f"请使用arxiv_search工具搜索学术论文: {query}"
        else:
            search_query = query

        result = await self.agent.run(search_query, deps=deps)
        return result.output

    def search_sync(
        self,
        query: str,
        deps: Optional[SearchDependencies] = None,
        search_type: Literal["comprehensive", "jina_search", "arxiv"] = "comprehensive",
    ) -> str:
        """同步执行搜索任务"""
        if deps is None:
            deps = SearchDependencies(
                jina_api_key=config.embedding.api_key,
                max_results=config.search.max_results,
                enable_content_extraction=config.search.enable_content_extraction,
            )

        # 根据搜索类型调整查询
        if search_type == "comprehensive":
            search_query = f"请使用comprehensive_search工具搜索: {query}"
        elif search_type == "jina_search":
            search_query = f"请使用jina_search工具搜索: {query}"
        elif search_type == "arxiv":
            search_query = f"请使用arxiv_search工具搜索学术论文: {query}"
        else:
            search_query = query

        result = self.agent.run_sync(search_query, deps=deps)
        return result.output

    async def extract_content(
        self,
        urls: List[str],
        deps: Optional[SearchDependencies] = None,
        analyze: bool = True,
    ) -> List[Dict[str, Any]]:
        """提取URLs内容的便利方法"""
        if deps is None:
            deps = SearchDependencies(
                jina_api_key=config.embedding.api_key,
                enable_content_extraction=config.search.enable_content_extraction,
            )

        if analyze:
            result = await self.agent.run(
                f"请使用extract_and_analyze_urls工具分析这些URL: {urls}", deps=deps
            )
        else:
            # 简单提取内容
            results = []
            for url in urls:
                content_result = await self.agent.run(
                    f"请使用jina_reader工具提取URL内容: {url}", deps=deps
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
    enable_content_extraction: bool = True,
) -> SearchDependencies:
    """创建搜索依赖配置"""
    return SearchDependencies(
        jina_api_key=jina_api_key or config.embedding.api_key,
        max_results=max_results,
        enable_content_extraction=enable_content_extraction,
    )


# 使用示例
async def main():
    """搜索代理使用示例"""
    info("CreatPartner 搜索代理演示")
    info("=" * 50)

    # 创建搜索代理
    agent = create_search_agent()

    # 创建依赖配置
    deps = create_search_dependencies(max_results=3, enable_content_extraction=True)

    # 演示不同类型的搜索
    queries = [
        ("AI在教育中的应用", "comprehensive"),
        ("machine learning in education", "arxiv"),
        ("人工智能教育趋势分析", "jina_search"),
    ]

    for query, search_type in queries:
        info(f"{search_type}搜索", query=query)
        try:
            result = await agent.search(query, deps, search_type)
            success("搜索完成")
            info("结果摘要", content=result[:200] + "...")
        except Exception as e:
            error("搜索失败", error=str(e))

    # 演示内容提取
    test_urls = [
        "https://jina.ai",
        "https://arxiv.org/abs/2301.00000",  # 示例URL
    ]

    info("内容提取演示", urls_count=len(test_urls))
    try:
        extraction_results = await agent.extract_content(test_urls, deps, analyze=True)
        success("内容提取完成")
        info("提取结果数量", count=len(extraction_results.get("results", [])))
    except Exception as e:
        error("内容提取失败", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
