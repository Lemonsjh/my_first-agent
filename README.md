# 扫地机器人智能客服

这是一个基于 FastAPI、LangChain Agent、Chroma 向量库和 DashScope 模型的扫地/扫拖机器人智能客服 Demo。项目支持登录注册、流式聊天、知识库检索、报告生成辅助工具和管理员知识库上传。

## 主要功能

- 智能客服对话：支持流式返回和 Markdown 渲染。
- RAG 检索：基于 Chroma 向量检索、BM25 关键词召回和 reranker。
- 对话历史：按登录用户保存历史，并在长对话时生成摘要。
- 知识库上传：管理员可上传 `txt` / `pdf` 文件并写入向量库。
- 工具调用：支持天气、时间、联网搜索和模拟外部使用记录查询。

## 环境准备

建议使用 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并设置必要环境变量：

```powershell
$env:FASTAPI_SECRET_KEY="replace-with-a-long-random-secret"
$env:DASHSCOPE_API_KEY="replace-with-your-dashscope-api-key"
$env:TAVILY_API_KEY="replace-with-your-tavily-api-key"
```

其中 `TAVILY_API_KEY` 只在调用联网搜索工具时需要。

## 启动

```powershell
python app.py
```

默认服务地址：

```text
http://localhost:5000
```

## 默认账号

当前 `users.json` 已使用 PBKDF2 哈希保存密码。已有演示账号仍可使用原密码登录：

- `admin` / `123456`
- `demo` / `demo123`

生产环境请删除默认账号或立即修改密码。

## 初始化知识库

知识库源文件位于 `data/`，配置在 `config/chroma.yml` 中。首次构建向量库时，可以临时取消 [app.py](D:/Codex001/react_agent_for_cleaner/app.py) 中 `VECTOR_STORE.load_document()` 的注释，或在 Python 交互环境中调用：

```python
from rag.vector_store import VectorStoreService

VectorStoreService().load_document()
```

## 注意事项

- 不要把 `.env`、日志、向量库、聊天记录提交到版本库。
- 线上部署必须设置稳定的 `FASTAPI_SECRET_KEY`，否则服务重启后登录态会失效。
- 当前用户系统适合 Demo，不建议直接作为生产账号体系使用。
