
# MIT License
# Copyright (c) 2024 Cursor AI

import time

from flask import Blueprint, jsonify

from app.core.vector_store import get_vector_store

health_bp = Blueprint("health", __name__, url_prefix="/health")
_start_time = time.time()


@health_bp.route("", methods=["GET"])
def health_check():
    store = get_vector_store()
    return jsonify({
        "status": "ok",
        "faiss_chunks": store.chunk_count(),
        "uptime_s": round(time.time() - _start_time, 2),
    }), 200
