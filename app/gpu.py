"""One big model on the GPU at a time: the reader (Qwen-VL) and the proofreader (Gemma) take turns.

On a 12 GB card the two don't fit together (~6.5 GB + ~8 GB), so whoever needs the GPU unloads the other.
HANDAPP_KEEP_BOTH=1 skips that on bigger cards.
"""

import os
import threading
import time
from contextlib import contextmanager

KEEP_BOTH = os.environ.get("HANDAPP_KEEP_BOTH", "0") == "1"

lock = threading.Lock()
_unloaders = {}  # model name -> function that frees it
_waiting = {}  # model name -> how many callers are queued for the lock
_waiting_lock = threading.Lock()
_last_used = {}  # model name -> when it last finished a job


def register(name: str, unload):
    _unloaders[name] = unload


def last_used(name: str) -> float:
    return _last_used.get(name, 0.0)


def waiting(name: str) -> int:
    """How many callers are waiting to use `name`; the proofreader yields to the reader when it's > 0."""
    return _waiting.get(name, 0)


@contextmanager
def use(name: str):
    """Hold the GPU for `name`, unloading the other models first."""
    with _waiting_lock:
        _waiting[name] = _waiting.get(name, 0) + 1
    try:
        lock.acquire()
    finally:
        with _waiting_lock:
            _waiting[name] -= 1
    try:
        if not KEEP_BOTH:
            for other, unload in _unloaders.items():
                if other != name:
                    unload()
        yield
    finally:
        _last_used[name] = time.time()
        lock.release()


def free_cache():
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
