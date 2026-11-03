"""`uv run doctor`: check this machine and the model server, then warm the chat model.

Run it after setup and again a minute before a demo.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import time

from . import llm
from .config import get_settings

GB = 1024**3


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")

    def warn(self, msg: str) -> None:
        print(f"  warn  {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        print(f"  FAIL  {msg}")

    def note(self, msg: str) -> None:
        print(f"        {msg}")


def total_memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout.strip()) / GB
        except (OSError, ValueError):
            pass
    return None


def available_memory_gb(total_gb: float) -> float | None:
    """Memory a model can use right now.

    macOS counts reclaimable cache as used, so ask `memory_pressure` for its free percentage
    (the same figure Activity Monitor's memory pressure is based on).
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
            m = re.search(r"free percentage:\s*(\d+)%", out.stdout)
            if m:
                return total_gb * int(m.group(1)) / 100
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"MemAvailable:\s*(\d+) kB", f.read())
            return int(m.group(1)) * 1024 / GB if m else None
    except OSError:
        return None


def recommended_chat_model(total_gb: float) -> str:
    # Machines report slightly under their marketed size (16 GB shows as ~15.6).
    if total_gb < 12:
        return "llama3.2:3b"
    if total_gb < 23:
        return "llama3.1:8b"
    return "llama3.1:8b (or a 14B model such as qwen2.5:14b)"


def check_machine(r: Report) -> None:
    print("\nMachine")
    # uv enforces requires-python (>=3.11) before this runs.
    r.ok(f"Python {platform.python_version()}")

    if sys.platform == "darwin" and platform.machine() == "arm64":
        r.ok("Apple Silicon (models run on the GPU)")
    elif sys.platform == "darwin":
        r.warn("Intel Mac: Ollama runs on the CPU only and will be slow")
    else:
        r.ok(f"{sys.platform}/{platform.machine()}")

    total = total_memory_gb()
    if total is None:
        r.warn("could not read total memory")
    else:
        if total >= 15:
            r.ok(f"{total:.0f} GB memory")
        elif total >= 7.5:
            r.warn(f"{total:.0f} GB memory: use CHAT_MODEL=llama3.2:3b")
        else:
            r.fail(f"{total:.0f} GB memory: at least 8 GB is needed")
        r.note(f"recommended chat model for this machine: {recommended_chat_model(total)}")
        free = available_memory_gb(total)
        if free is not None and free < 6:
            r.warn(
                f"only about {free:.1f} GB memory free right now: "
                "close heavy apps and browser tabs before a demo"
            )

    disk = shutil.disk_usage(".").free / GB
    if disk >= 10:
        r.ok(f"{disk:.0f} GB free disk")
    else:
        r.warn(f"{disk:.1f} GB free disk: models need about 6 GB")


def check_models(r: Report) -> bool:
    s = get_settings()
    print(f"\nModel server ({s.base_url})")
    try:
        available = llm.list_models()
    except llm.ModelServerError as err:
        r.fail(str(err))
        return False
    r.ok("reachable")
    ready = True
    for model in (s.embed_model, s.chat_model):
        if llm.has_model(available, model):
            r.ok(f"model {model} is pulled")
        else:
            r.fail(f"model {model} is missing: run `ollama pull {model}`")
            ready = False
    return ready


def smoke_test(r: Report) -> None:
    print("\nSmoke tests")
    try:
        t0 = time.perf_counter()
        [vector] = llm.embed(["The quick brown fox."], "document")
        r.ok(f"embeddings: {len(vector)} dimensions in {(time.perf_counter() - t0) * 1000:.0f} ms")
    except llm.ModelServerError as err:
        r.fail(f"embeddings: {err}")

    try:
        r.note("warming the chat model (the first load can take up to a minute)...")
        t0 = time.perf_counter()
        first_s = 0.0
        text = ""
        for token in llm.chat_stream(
            [{"role": "user", "content": "Reply with exactly one word: ready"}]
        ):
            if not first_s:
                first_s = time.perf_counter() - t0
            text += token
        total_s = time.perf_counter() - t0
        r.ok(f'chat: "{text.strip()[:40]}", first token {first_s:.1f}s, total {total_s:.1f}s')
    except llm.ModelServerError as err:
        r.fail(f"chat: {err}")


def main() -> None:
    r = Report()
    check_machine(r)
    if check_models(r):
        smoke_test(r)
    if r.failures:
        print(f"\n{r.failures} check(s) failed.\n")
        sys.exit(1)
    print("\nAll checks passed. Run `uv run serve`.\n")
