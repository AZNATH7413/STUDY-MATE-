from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
        "You are StudyGPT, a friendly, rigorous study coach. \n"
        "Priorities: (1) accuracy, (2) clarity, (3) brevity. \n"
        "Always reason step-by-step, check for misunderstandings, and adapt to the user's level. \n"
        "Use simple language and concrete examples when helpful. \n"
        "If the user asks about a graded problem, offer hints first and only give the full solution if they ask.\n"
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
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        headers = {"Content-Type": "application/json"}
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
            if r.status_code == 404:
                raise RuntimeError("Ollama endpoint not found. Is Ollama running?")
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
                    break


# ---------------- FastAPI app ----------------

app = FastAPI(title="Study AI Web")

static_dir = os.path.join(os.path.dirname(__file__), "web")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/health")
def api_health():
    provider = detect_provider()
    openai_model = getenv("STUDY_AI_OPENAI_MODEL", "gpt-4o-mini")
    ollama_model = getenv("STUDY_AI_OLLAMA_MODEL", "llama3.1:8b")
    return JSONResponse(
        {
            "provider": provider,
            "openai_model": openai_model,
            "ollama_model": ollama_model,
            "openai_key_present": bool(getenv("OPENAI_API_KEY")),
            "ollama_host": getenv("OLLAMA_HOST", "http://localhost:11434"),
        }
    )


@app.post("/api/chat")
def api_chat(req: Request):
    try:
        data = req.json()
    except Exception:
        # Fallback for sync Request with body already read by FastAPI under the hood
        data = None
    if data is None:
        data = requests.json.loads(req.body()) if hasattr(req, 'body') else {}

    # When using FastAPI sync endpoints, proper body parsing is via request.json() async.
    # But to keep things simple in a sync function, read from starlette's parsed body.
    try:
        data = data or {}
        if not data:
            data = __import__('json').loads(__import__('starlette').requests.Request(req.scope, req.receive))  # fallback noop
    except Exception:
        pass

    # Extract payload
    try:
        payload = data if isinstance(data, dict) else {}
    except Exception:
        payload = {}

    provider = payload.get("provider") or detect_provider()
    mode = payload.get("mode", "explain")
    user_messages = payload.get("messages", [])

    if not isinstance(user_messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    # Ensure system prompt is present as the first message
    messages: List[Dict[str, str]] = []
    if not user_messages or user_messages[0].get("role") != "system":
        messages.append({"role": "system", "content": system_prompt(mode)})
    messages.extend(user_messages)

    if provider == "openai":
        api_key = getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")
        model = getenv("STUDY_AI_OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAIStreamer(model, api_key)
    else:
        model = getenv("STUDY_AI_OLLAMA_MODEL", "llama3.1:8b")
        host = getenv("OLLAMA_HOST", "http://localhost:11434")
        client = OllamaStreamer(model, host=host)

    def generate():
        try:
            for piece in client.stream(messages):
                yield piece
        except Exception as e:
            # Send a minimal error; avoid leaking details
            yield f"\n[Error] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")  
