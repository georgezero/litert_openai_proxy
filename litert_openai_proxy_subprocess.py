#!/usr/bin/env python3
"""LiteRT OpenAI-compatible proxy.

Wraps `litert-lm run` as an OpenAI /v1/chat/completions endpoint.
Uses the correct Gemma chat-template format and supports SSE streaming
(fake-streaming: full response is buffered then chunked word-by-word).

True token-level streaming is not possible without a persistent inference
process — litert-lm v0.10.1 has no `serve` command.  The OS page-cache
keeps the 2.4 GB model file in RAM after the first call, so subsequent
calls are fast from a disk-I/O perspective; the overhead is Python process
startup + model deserialization each time.

Environment variables:
  LITERT_MODEL_ID   model id as shown by `litert-lm list`  (default: gemma-4-e2b-it)
  LITERT_BACKEND    cpu | gpu                               (default: cpu)
  OPENAI_API_KEY    if set, Bearer auth is required
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

DEFAULT_MODEL = os.environ.get("LITERT_MODEL_ID", "gemma-4-e2b-it")
LITERT_BACKEND = os.environ.get("LITERT_BACKEND", "cpu")
REQUIRE_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

app = FastAPI(title="LiteRT OpenAI Proxy")


# ---------------------------------------------------------------------------
# Request / response models
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
    token = authorization.split(" ", 1)[1].strip()
    if token != REQUIRE_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Gemma chat-template prompt builder
#
# Format: https://ai.google.dev/gemma/docs/formatting
#   <start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n…<end_of_turn>\n
#
# Gemma has no native system role.  A leading system message is injected as
# a user turn so the model sees the instruction before any real user content.
# ---------------------------------------------------------------------------


def _build_prompt(messages: List[Message]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.role.strip().lower()
        if role in {"system", "user", "human"}:
            parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>")
        elif role in {"assistant", "model"}:
            parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>")
        else:
            parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>")
    # Open the model turn for the response
    parts.append("<start_of_turn>model")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# litert-lm runner
# ---------------------------------------------------------------------------


def _litert_cmd(model_id: str, prompt: str) -> List[str]:
    return ["litert-lm", "run", model_id, "--prompt", prompt, "--backend", LITERT_BACKEND]


def _run_sync(model_id: str, prompt: str) -> subprocess.CompletedProcess:
    return subprocess.run(_litert_cmd(model_id, prompt), capture_output=True, text=True)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(cid: str, model: str, content: str, finish_reason: Optional[str]) -> str:
    delta = {"content": content} if content else {}
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _fake_stream(model_id: str, prompt: str, cid: str) -> AsyncIterator[str]:
    """Run litert-lm synchronously, then stream the response word-by-word as SSE.

    litert-lm has no serve/daemon mode so real per-token streaming is not
    available.  This approach at least satisfies clients that require
    Content-Type: text/event-stream without returning a 400.
    """
    loop = asyncio.get_event_loop()
    proc = await loop.run_in_executor(None, _run_sync, model_id, prompt)

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "litert-lm run failed"
        # Signal error in-stream then close
        yield _sse_chunk(cid, model_id, f"[ERROR: {err}]", "stop")
        yield "data: [DONE]\n\n"
        return

    content = proc.stdout.strip()
    words = content.split(" ")
    for i, word in enumerate(words):
        text = word + ("" if i == len(words) - 1 else " ")
        yield _sse_chunk(cid, model_id, text, None)
        await asyncio.sleep(0)  # yield to event loop between chunks

    yield _sse_chunk(cid, model_id, "", "stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": DEFAULT_MODEL, "backend": LITERT_BACKEND}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)) -> dict:
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model", "owned_by": "local-litert"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    if not body.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    model_id = body.model or DEFAULT_MODEL
    prompt = _build_prompt(body.messages)
    cid = f"chatcmpl-{uuid.uuid4().hex}"

    if body.stream:
        return StreamingResponse(
            _fake_stream(model_id, prompt, cid),
            media_type="text/event-stream",
        )

    # Non-streaming
    started = time.time()
    proc = _run_sync(model_id, prompt)
    elapsed = int((time.time() - started) * 1000)

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "litert-lm run failed"
        raise HTTPException(status_code=500, detail=detail)

    content = proc.stdout.strip()
    prompt_tokens = len(prompt.split())
    completion_tokens = len(content.split())

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_elapsed_ms": elapsed,
    }
