from __future__ import annotations

import io
import json
import os
import uuid
from typing import Dict, Iterator, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader

load_dotenv()

# ---------------- Config helpers ----------------

def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def detect_provider() -> str:
    explicit = getenv("STUDY_AI_PROVIDER")
    if explicit in {"openai", "ollama"}:
        return explicit
    if getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def system_prompt(mode: str = "explain") -> str:
    base = (
        "You are StudyGPT, a friendly, rigorous study coach. "
        "Priorities: (1) accuracy, (2) clarity, (3) brevity. "
        "Always reason step-by-step, check for misunderstandings, and adapt to the user's level. "
        "Use simple language and concrete examples when helpful. "
        "If the user asks about a graded problem, offer hints first and only give the full solution if they ask."
    )
    if mode == "quiz":
        base += (
            "\nMode: QUIZ. Ask one question at a time, wait for the student's answer, then provide concise feedback, the correct answer, and a short explanation. "
            "Adjust difficulty gradually and keep a supportive tone."
        )
    else:
        base += (
            "\nMode: EXPLAIN. Provide concise, step-by-step explanations and ask brief check-for-understanding questions."
        )
    return base


# ---------------- Streamers ----------------

class OpenAIStreamer:
    def __init__(self, model: str, api_key: str, temperature: float = 0.2):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
            if r.status_code == 401:
                raise RuntimeError("OpenAI auth failed (401). Ensure OPENAI_API_KEY is set.")
            if r.status_code == 429:
                raise RuntimeError("OpenAI rate limit (429). Try again shortly.")
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    data = raw[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content


class OllamaStreamer:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", temperature: float = 0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def _post_stream(self, host: str, messages: List[Dict[str, str]]):
        url = f"{host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        headers = {"Content-Type": "application/json"}
        return requests.post(url, headers=headers, json=payload, stream=True, timeout=600)

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        hosts_to_try = [self.host]
        if "localhost" in self.host:
            hosts_to_try.append(self.host.replace("localhost", "127.0.0.1"))
        else:
            hosts_to_try.append("http://127.0.0.1:11434")

        last_error_text = None
        for host in hosts_to_try:
            try:
                with self._post_stream(host, messages) as r:
                    if r.status_code == 404:
                        try:
                            err = r.json()
                            msg = err.get("error") or err.get("message")
                            last_error_text = msg
                        except Exception:
                            pass
                        continue
                    r.raise_for_status()
                    for raw in r.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        piece = obj.get("response")
                        if not piece:
                            msg = obj.get("message", {})
                            piece = msg.get("content")
                        if piece:
                            yield piece
                        if obj.get("done"):
                            return
            except Exception as e:
                last_error_text = str(e)
                continue
        raise RuntimeError(last_error_text or "Ollama endpoint not reachable. Is Ollama running?")


# ---------------- FastAPI app ----------------

app = FastAPI(title="Study AI Web")

BASE_DIR = os.path.dirname(__file__)
static_dir = os.path.join(BASE_DIR, "web")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CONTEXTS: Dict[str, str] = {}

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/health")
def api_health():
    provider = detect_provider()
    openai_model = getenv("STUDY_AI_OPENAI_MODEL", "gpt-4o-mini")
    ollama_model = getenv("STUDY_AI_OLLAMA_MODEL", "phi3:mini")
    return JSONResponse(
        {
            "provider": provider,
            "openai_model": openai_model,
            "ollama_model": ollama_model,
            "openai_key_present": bool(getenv("OPENAI_API_KEY")),
            "ollama_host": getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        }
    )


@app.post("/api/chat")
def api_chat(payload: Dict[str, object]):
    provider = (payload.get("provider") if isinstance(payload, dict) else None) or detect_provider()
    mode = (payload.get("mode") if isinstance(payload, dict) else None) or "explain"
    user_messages = (payload.get("messages") if isinstance(payload, dict) else None) or []
    context_id = (payload.get("context_id") if isinstance(payload, dict) else None)

    if not isinstance(user_messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    messages: List[Dict[str, str]] = []
    if not user_messages or user_messages[0].get("role") != "system":
        messages.append({"role": "system", "content": system_prompt(mode)})

    if context_id and isinstance(context_id, str):
        ctx_text = CONTEXTS.get(context_id)
        if not ctx_text:
            ctx_path = os.path.join(UPLOAD_DIR, f"{context_id}.txt")
            if os.path.exists(ctx_path):
                try:
                    with open(ctx_path, "r", encoding="utf-8", errors="ignore") as f:
                        ctx_text = f.read()
                        CONTEXTS[context_id] = ctx_text
                except Exception:
                    ctx_text = None
        if ctx_text:
            snippet = ctx_text[:120000]
            messages.append({
                "role": "system",
                "content": "Attached study context. Use it for answers; quote relevant parts.\n---\n" + snippet,
            })

    messages.extend(user_messages)

    if provider == "openai":
        api_key = getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")
        model = getenv("STUDY_AI_OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAIStreamer(model, api_key)
    else:
        model = getenv("STUDY_AI_OLLAMA_MODEL", "phi3:mini")
        host = getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = OllamaStreamer(model, host=host)

    def generate():
        try:
            for piece in client.stream(messages):
                yield piece
        except Exception as e:
            yield f"\n[Error] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    data = await file.read()
    filename = file.filename or "uploaded"
    name_lower = filename.lower()

    text = ""
    try:
        if name_lower.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n\n".join(pages)
        elif name_lower.endswith(".txt") or name_lower.endswith(".md"):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = data.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from file.")
    if len(text) > 500000:
        text = text[:500000]

    ctx_id = uuid.uuid4().hex
    CONTEXTS[ctx_id] = text

    out_path = os.path.join(UPLOAD_DIR, f"{ctx_id}.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

    return JSONResponse({
        "context_id": ctx_id,
        "filename": filename,
        "chars": len(text),
    })

