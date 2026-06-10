# MIT License
# Copyright (c) 2024 Cursor AI

import json
import requests
import structlog
from typing import List, Dict, Any, Optional, Generator
from app.core.config import Settings, get_settings

logger = structlog.get_logger()


def generate_answer(messages: List[Dict[str, str]], settings: Optional[Settings] = None) -> str:
    cfg = settings or get_settings()
    url = cfg.ollama_base_url + "/api/chat"
    model = cfg.ollama_llm_model
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7},
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def generate_answer_stream(messages: List[Dict[str, str]], settings: Optional[Settings] = None) -> Generator[str, None, None]:
    cfg = settings or get_settings()
    url = cfg.ollama_base_url + "/api/chat"
    model = cfg.ollama_llm_model
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7},
    }
    resp = requests.post(url, json=payload, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                yield chunk
        except Exception:
            continue