#!/usr/bin/env python3
"""LiteRT OpenAI-compatible proxy — native Python API.

Uses the litert_lm Python API to load the model once at startup and keep it
in memory.  Each request creates a fresh Conversation from the persistent
Engine.  send_message_async yields real tokens as they are generated — no
fake chunking.

Multi-turn history: prior turns are replayed through the Conversation (each
prior user turn runs inference and discards output to advance KV-cache state).
For the common single-turn case this is a no-op.  For long histories it adds
latency proportional to the number of prior turns.

Environment variables:
  LITERT_MODEL_ID   model id as shown by `litert-lm list`  (default: gemma-4-e2b-it)
  LITERT_BACKEND    cpu | gpu                               (default: cpu)
  OPENAI_API_KEY    if set, Bearer auth is required
  LOG_FILE          path to log file (default: ~/.local/state/litert-proxy/proxy.log)
"""

import asyncio
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

import litert_lm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("LITERT_MODEL_ID", "gemma-4-e2b-it")
MODEL_PATH = os.path.expanduser(
    f"~/.litert-lm/models/{MODEL_ID}/model.litertlm"
)
BACKEND = (
    litert_lm.Backend.GPU
    if os.environ.get("LITERT_BACKEND", "cpu").lower() == "gpu"
    else litert_lm.Backend.CPU
)
REQUIRE_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
LOG_FILE = os.path.expanduser(
    os.environ.get("LOG_FILE", "~/.local/state/litert-proxy/proxy.log")
)

# ---------------------------------------------------------------------------
# Logging — file (rotating 10 MB × 3) + stdout
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3
)
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_fmt)

log = logging.getLogger("litert_proxy")
log.setLevel(logging.INFO)
log.addHandler(_file_handler)
log.addHandler(_stream_handler)

# ---------------------------------------------------------------------------
# Engine singleton — loaded once, lives for the process lifetime
# ---------------------------------------------------------------------------

_engine: Optional[litert_lm.Engine] = None
# Only one conversation may run at a time — litert_lm Engine/Conversation is
# not thread-safe.  Requests that cannot acquire the lock within the timeout
# receive a 503 rather than crashing the engine.
_inference_lock = threading.Lock()
INFERENCE_LOCK_TIMEOUT = 300  # seconds — matches client request timeout
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))
# Wall-clock timeout per inference request. cancel_process() is called when
# this expires, then the thread exits and releases the lock.
INFERENCE_TIMEOUT_SECS = int(os.environ.get("INFERENCE_TIMEOUT_SECS", 120))


def _suppress_native_logs() -> None:
    """Redirect C++ fd 1/2 to /dev/null; keep Python stdout/stderr."""
    try:
        orig_out = os.dup(sys.stdout.fileno())
        orig_err = os.dup(sys.stderr.fileno())
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        sys.stdout = open(orig_out, "w", closefd=False)
        sys.stderr = open(orig_err, "w", closefd=False)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _suppress_native_logs()
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}\n"
            f"Run: litert-lm import --from-huggingface-repo "
            f"litert-community/gemma-4-E2B-it-litert-lm "
            f"gemma-4-E2B-it.litertlm {MODEL_ID}"
        )

    log.info(f"Loading model: {MODEL_PATH} (backend={BACKEND})")
    _engine = litert_lm.Engine(MODEL_PATH, backend=BACKEND)
    _engine.__enter__()
    log.info(f"Model loaded and ready. Logging to {LOG_FILE}")
    yield
    _engine.__exit__(None, None, None)
    _engine = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LiteRT Native Proxy", lifespan=lifespan)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    elapsed = (time.time() - started) * 1000
    log.info(
        f"{request.method} {request.url.path} → {response.status_code} {elapsed:.0f}ms"
    )
    return response


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(authorization: Optional[str]) -> None:
    if not REQUIRE_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.split(" ", 1)[1].strip() != REQUIRE_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(cid: str, model: str, text: str, finish_reason: Optional[str]) -> str:
    delta = {"content": text} if text else {}
    return "data: " + json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }) + "\n\n"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _build_system_init(messages: List[Message]) -> List[dict]:
    """Return litert_lm-format system messages for create_conversation."""
    system_texts = [m.content for m in messages if m.role == "system"]
    if not system_texts:
        return []
    return [{"role": "system", "content": [{"type": "text", "text": t}]}
            for t in system_texts]


def _extract_text(chunk: dict) -> str:
    """Pull text out of a send_message_async chunk."""
    for item in chunk.get("content", []):
        if item.get("type") == "text":
            return item.get("text", "")
    return ""


def _run_conversation_in_thread(
    messages: List[Message],
    result_queue: "queue.Queue[str | None | Exception]",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    """Run the full conversation in a background thread, pushing tokens to queue."""
    if not _inference_lock.acquire(timeout=INFERENCE_LOCK_TIMEOUT):
        result_queue.put(RuntimeError("inference lock timeout — another request is still running"))
        result_queue.put(None)
        return
    try:
        system_init = _build_system_init(messages)
        non_system = [m for m in messages if m.role != "system"]

        if not non_system:
            result_queue.put(None)
            return

        prior_turns = non_system[:-1]   # all but last
        last_msg = non_system[-1]

        with _engine.create_conversation(
            messages=system_init if system_init else None
        ) as conv:
            # Replay prior turns to advance KV-cache (no output kept).
            for m in prior_turns:
                if m.role in ("user", "human"):
                    for _ in conv.send_message_async(m.content):
                        pass

            # Timeout watchdog: cancel after INFERENCE_TIMEOUT_SECS wall time.
            def _watchdog():
                log.warning(f"Inference timeout after {INFERENCE_TIMEOUT_SECS}s — cancelling")
                try:
                    conv.cancel_process()
                except Exception:
                    pass

            watchdog = threading.Timer(INFERENCE_TIMEOUT_SECS, _watchdog)
            watchdog.start()
            try:
                # Stream the final user message token by token.
                token_count = 0
                recent: list[str] = []
                REPEAT_WINDOW = 20  # cancel if last 20 tokens are identical
                for chunk in conv.send_message_async(last_msg.content):
                    text = _extract_text(chunk)
                    if text:
                        result_queue.put(text)
                        token_count += 1
                        if token_count >= max_tokens:
                            log.warning("max_tokens reached — cancelling")
                            conv.cancel_process()
                            break
                        recent.append(text)
                        if len(recent) > REPEAT_WINDOW:
                            recent.pop(0)
                        if len(recent) == REPEAT_WINDOW and len(set(recent)) == 1:
                            log.warning(f"Repetition loop detected — cancelling")
                            conv.cancel_process()
                            break
            finally:
                watchdog.cancel()

    except Exception as exc:
        result_queue.put(exc)
    finally:
        _inference_lock.release()
        result_queue.put(None)  # sentinel


async def _stream_sse(
    messages: List[Message], cid: str, model_id: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> AsyncIterator[str]:
    q: "queue.Queue[str | None | Exception]" = queue.Queue()
    thread = threading.Thread(
        target=_run_conversation_in_thread,
        args=(messages, q, max_tokens),
        daemon=True,
    )
    thread.start()

    loop = asyncio.get_event_loop()

    parts: List[str] = []
    while True:
        # Fetch next item without blocking the event loop
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            yield _sse_chunk(cid, model_id, f"[ERROR: {item}]", "stop")
            yield "data: [DONE]\n\n"
            return
        parts.append(item)
        yield _sse_chunk(cid, model_id, item, None)

    log.info(f"<< [assistant] {''.join(parts)}")
    yield _sse_chunk(cid, model_id, "", "stop")
    yield "data: [DONE]\n\n"


async def _run_sync(messages: List[Message]) -> str:
    """Non-streaming path: run inference in executor, return full text."""
    q: "queue.Queue[str | None | Exception]" = queue.Queue()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _run_conversation_in_thread, messages, q
    )
    parts: List[str] = []
    while True:
        item = q.get_nowait() if not q.empty() else None
        if item is None:
            break
        if isinstance(item, Exception):
            raise RuntimeError(str(item))
        parts.append(item)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "model": MODEL_ID,
        "backend": "gpu" if BACKEND == litert_lm.Backend.GPU else "cpu",
        "engine_loaded": _engine is not None,
    }


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)) -> dict:
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-litert"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    if not body.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    model_id = body.model or MODEL_ID
    cid = f"chatcmpl-{uuid.uuid4().hex}"

    max_tokens = body.max_tokens if body.max_tokens else DEFAULT_MAX_TOKENS

    # Log incoming messages
    for m in body.messages:
        log.info(f">> [{m.role}] {m.content}")

    if body.stream:
        return StreamingResponse(
            _stream_sse(body.messages, cid, model_id, max_tokens),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full response
    started = time.time()
    q: "queue.Queue[str | None | Exception]" = queue.Queue()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _run_conversation_in_thread, body.messages, q, max_tokens
    )
    parts: List[str] = []
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise HTTPException(status_code=500, detail=str(item))
        parts.append(item)

    content = "".join(parts)
    elapsed = int((time.time() - started) * 1000)
    log.info(f"<< [assistant] {content}")

    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(started),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in body.messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.content.split()) for m in body.messages) + len(content.split()),
        },
        "x_elapsed_ms": elapsed,
    }
