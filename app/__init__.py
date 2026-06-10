from flask import Flask
from flask_cors import CORS

from app.api import health_bp, ingest_bp, query_bp
from app.core.config import get_settings


def create_app():
    app = Flask(__name__)
    CORS(app)
    get_settings()
    app.register_blueprint(health_bp)
    app.register_blueprint(ingest_bp)
    app.register_blueprint(query_bp)
    return app
