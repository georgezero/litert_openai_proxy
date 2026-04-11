#!/usr/bin/env python3
"""Eval: compare subprocess proxy vs native proxy.

Runs the same 10 prompts against both proxies and reports timing.

Usage:
    # New proxy must already be running on PORT_NEW (default 8000)
    # Script starts the old subprocess proxy on PORT_OLD (default 8001)
    python3 eval.py

    # Override ports
    PORT_NEW=8000 PORT_OLD=8001 python3 eval.py

    # Skip starting old proxy (if already running externally)
    SKIP_START_OLD=1 python3 eval.py
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Optional

PORT_NEW = int(os.environ.get("PORT_NEW", 8000))
PORT_OLD = int(os.environ.get("PORT_OLD", 8001))
SKIP_START_OLD = os.environ.get("SKIP_START_OLD", "")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

PROMPTS = [
    # (label, messages) — kept short to complete within ~60s on Pi 5
    ("factual-short",
     [{"role": "user", "content": "What is the capital of France? One word answer."}]),

    ("factual-medium",
     [{"role": "user", "content": "What is DNS? Answer in exactly one sentence."}]),

    ("list-short",
     [{"role": "user", "content": "Name 3 Python built-in functions. One line each, no explanation."}]),

    ("reasoning",
     [{"role": "user", "content": "Why is TCP slower than UDP? Answer in one sentence."}]),

    ("code-short",
     [{"role": "user", "content": "Write a one-line Python function that reverses a string."}]),

    ("math",
     [{"role": "user", "content": "A train travels 120 km in 90 minutes. What is its speed in km/h? Just the number."}]),

    ("system-prompt",
     [{"role": "system", "content": "Reply in one sentence only."},
      {"role": "user", "content": "What is the difference between RAM and storage?"}]),

    ("creative-short",
     [{"role": "user", "content": "Write a haiku about a Raspberry Pi."}]),

    ("summarise",
     [{"role": "user", "content": (
         "Summarise in one sentence: "
         "TCP uses a three-way handshake (SYN, SYN-ACK, ACK) to establish a reliable "
         "connection before sending data, whereas UDP sends data immediately without handshaking."
     )}]),

    ("yesno",
     [{"role": "user", "content": "Is Python interpreted? Yes or no."}]),
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def chat(port: int, messages: list, timeout: float = 300.0) -> tuple[str, float]:
    """Returns (response_text, elapsed_seconds)."""
    payload = json.dumps({
        "model": "gemma-4-e2b-it",
        "messages": messages,
        "stream": False,
    }).encode()
    hdrs = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY") and port == PORT_NEW:
        hdrs["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers=hdrs,
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    elapsed = time.time() - t0
    text = body["choices"][0]["message"]["content"]
    return text, elapsed


# ---------------------------------------------------------------------------
# Start / stop old subprocess proxy
# ---------------------------------------------------------------------------

_old_proc: Optional[subprocess.Popen] = None


def start_old_proxy() -> bool:
    global _old_proc
    env = os.environ.copy()
    env.update({
        "PORT": str(PORT_OLD),
        "MODEL_ID": "gemma-4-e2b-it",
        "LITERT_MODEL_ID": "gemma-4-e2b-it",
        "VENV_PATH": os.path.expanduser("~/.venvs/litert-openai-proxy"),
        "APP_PATH": os.path.join(REPO_DIR, "old", "litert_openai_proxy_subprocess.py"),
        "OPENAI_API_KEY": "",  # don't inherit caller's OpenAI key
    })
    script = os.path.join(REPO_DIR, "old", "run-litert-openai-proxy-subprocess.sh")
    print(f"  Starting subprocess proxy on port {PORT_OLD}...")
    _old_proc = subprocess.Popen(
        [script],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return wait_for_port(PORT_OLD, timeout=30)


def stop_old_proxy():
    global _old_proc
    if _old_proc:
        os.killpg(os.getpgid(_old_proc.pid), signal.SIGTERM)
        _old_proc = None
        print(f"  Subprocess proxy stopped.")


# ---------------------------------------------------------------------------
# Run eval
# ---------------------------------------------------------------------------

def run_suite(port: int, label: str) -> list[dict]:
    results = []
    for name, messages in PROMPTS:
        # Wait for port to be healthy before each prompt
        if not wait_for_port(port, timeout=15):
            results.append({"prompt": name, "elapsed": None, "ok": False, "error": "port not healthy"})
            print(f"    {name:<20} SKIP (port unhealthy)")
            continue
        try:
            text, elapsed = chat(port, messages)
            words = len(text.split())
            wps = words / elapsed if elapsed > 0 else 0
            results.append({
                "prompt": name,
                "elapsed": elapsed,
                "words": words,
                "words_per_sec": wps,
                "ok": True,
            })
            print(f"    {name:<20} {elapsed:5.1f}s  {words:4d}w  {wps:5.1f}w/s")
        except Exception as e:
            results.append({"prompt": name, "elapsed": None, "ok": False, "error": str(e)})
            print(f"    {name:<20} ERROR: {e}")
            time.sleep(3)  # brief pause to let service recover if it restarted
    return results


def print_summary(new_results: list[dict], old_results: list[dict]):
    print("\n" + "=" * 65)
    print(f"{'PROMPT':<20} {'NEW(s)':>7} {'OLD(s)':>7} {'DIFF':>7} {'FASTER':>7}")
    print("-" * 65)
    new_total = old_total = 0.0
    for n, o in zip(new_results, old_results):
        if not n["ok"] or not o["ok"]:
            print(f"{n['prompt']:<20}   ERROR")
            continue
        diff = o["elapsed"] - n["elapsed"]
        pct = (diff / o["elapsed"]) * 100 if o["elapsed"] else 0
        tag = f"{pct:+.0f}%" if diff != 0 else "same"
        print(f"{n['prompt']:<20} {n['elapsed']:>7.1f} {o['elapsed']:>7.1f} {diff:>+7.1f} {tag:>7}")
        new_total += n["elapsed"]
        old_total += o["elapsed"]
    print("-" * 65)
    total_diff = old_total - new_total
    total_pct = (total_diff / old_total * 100) if old_total else 0
    print(f"{'TOTAL':<20} {new_total:>7.1f} {old_total:>7.1f} {total_diff:>+7.1f} {total_pct:>+6.0f}%")
    print(f"{'AVERAGE':<20} {new_total/len(new_results):>7.1f} {old_total/len(old_results):>7.1f}")
    print("=" * 65)
    print("\nnew = native litert_lm Python API (model in memory)")
    print("old = subprocess litert-lm run (process per request)")


def stop_new_proxy() -> None:
    print("  Stopping native proxy service...")
    subprocess.run(["systemctl", "--user", "stop", "litert-openai-proxy"],
                   capture_output=True)
    time.sleep(2)


def restart_new_proxy() -> None:
    print("  Restarting native proxy service...")
    subprocess.run(["systemctl", "--user", "start", "litert-openai-proxy"],
                   capture_output=True)
    wait_for_port(PORT_NEW, timeout=30)


def main():
    print(f"\n{'='*65}")
    print(f"litert-lm proxy eval — {len(PROMPTS)} prompts")
    print(f"NOTE: proxies run sequentially to avoid memory pressure")
    print(f"{'='*65}\n")

    # 1. Native proxy (service must be running)
    print(f"[1/2] Native proxy (port {PORT_NEW})")
    if not wait_for_port(PORT_NEW, timeout=5):
        print(f"  ERROR: new proxy not responding. Start with: systemctl --user start litert-openai-proxy")
        sys.exit(1)
    print(f"  Ready. Running {len(PROMPTS)} prompts...")
    new_results = run_suite(PORT_NEW, "native")

    # Stop native proxy before starting subprocess proxy — frees the 2.4 GB model
    stop_new_proxy()

    # 2. Subprocess proxy
    print(f"\n[2/2] Subprocess proxy (port {PORT_OLD})")
    if not SKIP_START_OLD:
        if not start_old_proxy():
            print(f"  ERROR: subprocess proxy failed to start")
            restart_new_proxy()
            stop_old_proxy()
            sys.exit(1)
    else:
        if not wait_for_port(PORT_OLD, timeout=5):
            print(f"  ERROR: subprocess proxy not responding on port {PORT_OLD}")
            restart_new_proxy()
            sys.exit(1)

    try:
        print(f"  Ready. Running {len(PROMPTS)} prompts...")
        old_results = run_suite(PORT_OLD, "subprocess")
    finally:
        if not SKIP_START_OLD:
            stop_old_proxy()
        restart_new_proxy()

    print_summary(new_results, old_results)


if __name__ == "__main__":
    main()
