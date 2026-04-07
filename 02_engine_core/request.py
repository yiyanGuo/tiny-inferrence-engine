from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional
import time

class RequestStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    ABORTED = auto()

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 64
    eos_token_id: Optional[int] = None

@dataclass
class RequestRuntime:
    kv_cache_handle: Optional[Any] = None
    prefill_done: bool = False
    last_token_id: Optional[int] = None
    scheduled_steps: int = 0
    cache_tokens: int = 0

@dataclass
class Request:
    request_id: str
    inputs: Optional[dict] = None
    request_status: RequestStatus = None