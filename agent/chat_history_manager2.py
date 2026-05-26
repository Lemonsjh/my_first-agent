import json
import threading
from typing import Any

from utils.path_tool import get_abs_path
from utils.logger_handler import logger

HISTORY_FILE = get_abs_path("chat_history2.json")
MAX_HISTORY_LENGTH = 10


class ChatHistoryManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.user_history = self._load_history()

    def _load_history(self) -> dict[str, Any]:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[ChatHistoryManager] 加载历史记录失败: {exc}")
            return {}

        if not isinstance(data, dict):
            return {}

        # 兼容旧格式：{user_id: [messages...]}
        for user_id, content in list(data.items()):
            if isinstance(content, list):
                data[user_id] = {"summary": "", "messages": content}

        return data

    def _save_history(self) -> None:
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.user_history, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error(f"[ChatHistoryManager] 保存历史记录失败: {exc}")

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self.user_history:
            self.user_history[user_id] = {"summary": "", "messages": []}

    def get_messages(self, user_id: str = "default_user") -> list[dict[str, str]]:
        with self._lock:
            user_data = self.user_history.get(user_id, {"summary": "", "messages": []})
            return list(user_data.get("messages", []))

    def get_summary(self, user_id: str = "default_user") -> str:
        with self._lock:
            user_data = self.user_history.get(user_id, {"summary": "", "messages": []})
            return str(user_data.get("summary", ""))

    def update_summary(self, summary_text: str, user_id: str = "default_user") -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.user_history[user_id]["summary"] = summary_text
            self._save_history()

    def set_messages(self, messages: list[dict[str, str]], user_id: str = "default_user") -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.user_history[user_id]["messages"] = list(messages)
            self._save_history()

    def add_user_message(self, query: str, user_id: str = "default_user") -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.user_history[user_id]["messages"].append({"role": "user", "content": query})

            # 防止历史无限增长
            if len(self.user_history[user_id]["messages"]) > MAX_HISTORY_LENGTH * 2:
                self.user_history[user_id]["messages"] = self.user_history[user_id]["messages"][-MAX_HISTORY_LENGTH:]

            # 用户消息也立刻落盘，避免中途异常导致数据丢失
            self._save_history()

    def add_ai_message(self, response: str, user_id: str = "default_user") -> None:
        with self._lock:
            self._ensure_user(user_id)
            self.user_history[user_id]["messages"].append({"role": "assistant", "content": response})
            self._save_history()

    def clear_history(self, user_id: str = "default_user") -> None:
        with self._lock:
            self.user_history[user_id] = {"summary": "", "messages": []}
            self._save_history()
