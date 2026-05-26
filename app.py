import json
import os
import secrets
import hashlib
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from agent.react_agent2 import ReactAgent
from rag.vector_store import VectorStoreService
import uvicorn
from utils.config_handler import chroma_conf

USERS_FILE = BASE_DIR / "users.json"
DATA_DIR = BASE_DIR / chroma_conf["data_path"]
ADMIN_USERNAMES = {"admin", "sjh"}
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260000


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _is_password_hash(value: str) -> bool:
    return value.startswith(f"{PASSWORD_SCHEME}$")


def _verify_password(password: str, stored_value: str) -> bool:
    if not _is_password_hash(stored_value):
        return secrets.compare_digest(password, stored_value)

    try:
        scheme, iterations, salt_hex, digest_hex = stored_value.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return secrets.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("FASTAPI_SECRET_KEY", secrets.token_hex(16)),
    max_age=int(timedelta(days=3).total_seconds()),
    same_site="lax",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 初始化向量存储和 Agent（注意：这两个服务现在都是单例模式）
VECTOR_STORE = VectorStoreService()
AGENT = ReactAgent()

# 首次部署时，取消下面这行注释来加载知识库到向量库：
# VECTOR_STORE.load_document()
DEFAULT_GREETING = {"role": "assistant", "content": "你好，我是扫地机器人智能客服，请问有什么可以帮你？"}


def _load_users() -> dict[str, str]:
    users: dict[str, str] = {}
    if USERS_FILE.exists():
        try:
            data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                users = {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    else:
        users = {"admin": "123456", "demo": "demo123"}

    needs_migration = False
    for username, stored_value in list(users.items()):
        if not _is_password_hash(stored_value):
            users[username] = _hash_password(stored_value)
            needs_migration = True

    if needs_migration:
        _save_users(users)

    return users


def _save_users(users: dict[str, str]) -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


USERS = _load_users()


def _is_logged_in(request: Request) -> bool:
    return bool(request.session.get("username"))


def _require_login_page(request: Request):
    if not _is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    return None


def _require_login_api(request: Request):
    if not _is_logged_in(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _is_admin(request: Request) -> bool:
    return request.session.get("username") in ADMIN_USERNAMES


def _require_admin_api(request: Request):
    unauth = _require_login_api(request)
    if unauth:
        return unauth
    if not _is_admin(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


def get_user_id(request: Request) -> str:
    username = request.session.get("username")
    if username:
        return f"user_{username}"

    user_id = request.session.get("user_id")
    if not user_id:
        user_id = f"user_{secrets.token_hex(8)}"
        request.session["user_id"] = user_id
    return user_id


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _is_logged_in(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    stored_password = USERS.get(username)
    if not stored_password or not _verify_password(password, stored_password):
        return templates.TemplateResponse(request, "login.html", {"error": "用户名或密码错误"})

    request.session.clear()
    request.session["username"] = username
    return RedirectResponse(url="/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if _is_logged_in(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"error": ""})


@app.post("/register")
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = username.strip()

    if not username or not password:
        return templates.TemplateResponse(request, "register.html", {"error": "用户名和密码不能为空"})
    if username in USERS:
        return templates.TemplateResponse(request, "register.html", {"error": "用户名已存在"})
    if password != confirm_password:
        return templates.TemplateResponse(request, "register.html", {"error": "两次密码不一致"})

    USERS[username] = _hash_password(password)
    _save_users(USERS)
    return RedirectResponse(url="/login", status_code=302)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    unauth = _require_login_page(request)
    if unauth:
        return unauth
    return templates.TemplateResponse(
        request,
        "index.html",
        {"username": request.session.get("username", ""), "is_admin": _is_admin(request)},
    )


@app.get("/api/history")
def get_history(request: Request):
    unauth = _require_login_api(request)
    if unauth:
        return unauth

    user_id = get_user_id(request)
    history = AGENT.history_manager.get_messages(user_id=user_id)
    if not history:
        history = [DEFAULT_GREETING]
    return {"messages": history}


@app.post("/api/clear")
def clear_history(request: Request):
    unauth = _require_login_api(request)
    if unauth:
        return unauth

    user_id = get_user_id(request)
    AGENT.history_manager.clear_history(user_id=user_id)
    return {"ok": True}


@app.post("/api/chat")
async def chat(request: Request):
    unauth = _require_login_api(request)
    if unauth:
        return unauth

    payload = await request.json()
    prompt = (payload.get("message") or "").strip()
    if not prompt:
        return JSONResponse({"error": "message is required"}, status_code=400)

    user_id = get_user_id(request)

    def generate():
        try:
            for chunk in AGENT.execute_stream(prompt, user_id=user_id):
                yield chunk
        except Exception as exc:
            yield f"\n\n[系统提示] 抱歉，处理请求时发生错误：{exc}"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/knowledge/upload")
async def upload_knowledge(request: Request, file: UploadFile = File(...)):
    admin_check = _require_admin_api(request)
    if admin_check:
        return admin_check

    filename = Path(file.filename or "").name.strip()
    if not filename:
        return JSONResponse({"error": "file is required"}, status_code=400)

    allowed_types = {ext.lower().lstrip(".") for ext in chroma_conf["allow_knowledge_file_type"]}
    file_ext = Path(filename).suffix.lower().lstrip(".")
    if file_ext not in allowed_types:
        return JSONResponse({"error": f"unsupported file type: .{file_ext}"}, status_code=400)

    target_path = DATA_DIR / filename
    if target_path.exists():
        return JSONResponse({"error": "file already exists"}, status_code=400)

    file_bytes = await file.read()
    if not file_bytes:
        return JSONResponse({"error": "empty file"}, status_code=400)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(file_bytes)

    try:
        ingest_result = VECTOR_STORE.ingest_file(str(target_path))
    except Exception as exc:
        return JSONResponse({"error": f"ingest failed: {exc}"}, status_code=500)

    return {
        "ok": True,
        "filename": filename,
        "ingested": bool(ingest_result.get("ingested")),
        "message": "知识库文件上传成功并已处理",
        "result": ingest_result,
    }


if __name__ == "__main__":
    # ========== 选择启动模式 ==========
    # 模式 1：开发模式（支持代码热重载，会启动两个进程）
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
    
    # 模式 2：生产模式（只启动一个进程，启动更快）
    # uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
