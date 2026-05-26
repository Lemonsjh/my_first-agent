import os
import asyncio
import csv
import json
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
import requests
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from tavily import TavilyClient
from datetime import datetime
rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    try:
        # 由于工具函数是同步的，使用 asyncio.run 来运行异步方法
        import nest_asyncio
        try:
            nest_asyncio.apply()
        except:
            pass
        reg_result = asyncio.run(rag.rag_summarize(query))
        return reg_result
    except Exception as e:
        logger.exception(f"rag_summarize 执行失败: {e}")
        return f" [系统提示：检索暂不可用，原因：{str(e)}] "


# 异步版本的工具，供 Agent 内部使用
async def rag_summarize_async(query: str) -> str:
    """异步版本的 RAG 检索工具"""
    try:
        return await rag.rag_summarize(query)
    except Exception as e:
        logger.exception(f"rag_summarize_async 执行失败: {e}")
        return f" [系统提示：检索暂不可用，原因：{str(e)}] "


@tool(description="根据城市名称查询中文简体实时天气")
def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=城市:%l,天气:%C,温度:%t,风向风力:%w,湿度:%h"
        res = requests.get(url, timeout=10)
        return res.text.strip()
    except Exception as e:
        return f"天气查询异常：{e}"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")

@tool(description="获取系统当前完整时间，格式为：年-月-日 时:分")
def get_current_time() ->str:
    """获取当前系统北京时间，返回格式化时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")

@tool(description="联网搜索内容，以字符串形式返回")
def search(query: str) -> str:
    """
    一个基于Tavily的网页搜索引擎工具。
    """
    print(f"🔍 正在执行 [Tavily] 网页搜索: {query}")
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "错误：TAVILY_API_KEY 未配置，请在环境变量中设置后再使用联网搜索。"

        client = TavilyClient(api_key)
        response = client.search(
            query=query,
            search_depth="advanced"
        )

        # 智能解析:优先返回直接答案
        if response.get("results"):
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('content', '')}"
                for i, res in enumerate(response["results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get("用户ID", "").strip()
                feature = row.get("特征", "").strip()
                efficiency = row.get("清洁效率", "").strip()
                consumables = row.get("耗材", "").strip()
                comparison = row.get("对比", "").strip()
                time = row.get("时间", "").strip()

                if not user_id or not time:
                    continue

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
