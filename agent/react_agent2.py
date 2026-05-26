from langchain.agents import create_agent

from agent.chat_history_manager2 import ChatHistoryManager
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_current_time,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
    search,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts

SUMMARY_TRIGGER_LENGTH = 10
SUMMARY_BATCH_SIZE = 6


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                get_current_time,
                fetch_external_data,
                fill_context_for_report,
                search
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        self.history_manager = ChatHistoryManager()

    def _generate_summary(self, old_summary: str, new_messages: list[dict[str, str]]) -> str:
        prompt = (
            f"当前摘要: {old_summary}\n"
            f"新的对话内容: {new_messages}\n"
            "请根据以上内容更新摘要，简要保留关键信息（如姓名、偏好、核心诉求）。"
        )
        response = chat_model.invoke(prompt)
        return str(response.content)

    def _maybe_summarize_history(self, user_id: str) -> str:
        current_summary = self.history_manager.get_summary(user_id)
        messages = self.history_manager.get_messages(user_id)

        if len(messages) <= SUMMARY_TRIGGER_LENGTH:
            return current_summary

        to_summarize = messages[:SUMMARY_BATCH_SIZE]
        remaining = messages[SUMMARY_BATCH_SIZE:]
        new_summary = self._generate_summary(current_summary, to_summarize)

        self.history_manager.update_summary(summary_text=new_summary, user_id=user_id)
        self.history_manager.set_messages(remaining, user_id)

        return new_summary

    def execute_stream(self, query: str, user_id: str):
        current_summary = self._maybe_summarize_history(user_id)

        # 先落盘用户输入，确保服务中断也不丢用户消息
        self.history_manager.add_user_message(query, user_id)

        input_messages = []
        if current_summary:
            input_messages.append({"role": "system", "content": f"这是之前的对话摘要: {current_summary}"})
        input_messages.extend(self.history_manager.get_messages(user_id))

        input_dict = {"messages": input_messages}
        last_full_content = ""

        # try:
        #     for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
        #         messages = chunk.get("messages", [])
        #         if not messages: continue
        #
        #         latest_msg = messages[-1]
        #         # 确保只处理 AI 的输出
        #         if getattr(latest_msg, "type", "") == "ai":
        #             content = latest_msg.content
        #             if content and len(content) > len(last_full_content):
        #                 new_piece = content[len(last_full_content):]
        #                 yield new_piece
        #                 last_full_content = content
        #
        # except Exception:
        #     logger.exception("[ReactAgent] 流式执行失败")
        #     raise
        # finally:
        #     if last_full_content:
        #         self.history_manager.add_ai_message(last_full_content, user_id)

        # 引入一个状态锁，确保“正在检索”的提示语只对前端输出一次
        has_yielded_status = False
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="messages", context={"report": False}):
                if isinstance(chunk, tuple) and len(chunk) > 0:
                    msg = chunk[0]

                    # 调试：如果还是不出，建议取消下面这行注释看控制台输出的 type 到底是什么
                    # print(f"Type: {getattr(msg, 'type', 'None')}, Content: {getattr(msg, 'content', '')}")

                    # 1. 排除明确的工具返回结果消息 (ToolMessage)
                    if getattr(msg, "type", "") == "tool":
                        yield " "  # 维持连接，但不显示搜索结果原文
                        continue

                    # 2. 拦截 AI 的工具调用指令 (Tool Calls) 并进行状态感知
                    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                        # 如果还没有输出过状态提示，则深入检查调用的工具名
                        if not has_yielded_status:
                            for tool_call in msg.tool_calls:
                                if tool_call.get("name") == "rag_summarize":
                                    yield "🔍 [正在检索知识库...] \n"
                                    has_yielded_status = True  # 锁死状态，防止流式块重复触发
                                    break
                                else:
                                    yield "🔍 [工具调用中...] \n"
                                    has_yielded_status = True  # 锁死状态，防止流式块重复触发
                                    break

                        continue

                    # 3. 只要有文本内容，且不是上述排除的情况，就认为是 AI 的回答
                    if hasattr(msg, "content"):
                        content_piece = msg.content
                        if content_piece and isinstance(content_piece, str):
                            yield content_piece
                            last_full_content += content_piece


        except Exception:
            logger.exception("[ReactAgent] 流式执行失败")
            raise
        finally:
            # 将完整回答存入历史
            if last_full_content:
                self.history_manager.add_ai_message(last_full_content, user_id)


if __name__ == "__main__":
    agent = ReactAgent()

    # for chunk in agent.execute_stream("我叫张三", user_id="user1"):
    #     print(chunk, end="", flush=True)
    #
    for chunk in agent.execute_stream("你好？", user_id="user1"):
        print(chunk, end="", flush=True)
    # for chunk in agent.execute_stream("今天xiaomi公司股价如何？", user_id="user1"):
    #     print(chunk, end="", flush=True)

