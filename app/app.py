from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .rag import RagPipeline
from .database import create_database_with_spec

def create_app() -> Flask:
    app = Flask(__name__)

    corpus_path = os.getenv("RAG_CORPUS_PATH", "data/rag_corpus.json")
    app.rag_pipeline = RagPipeline(corpus_path=corpus_path)  # type: ignore[attr-defined]

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    database, function_spec = create_database_with_spec(data_dir=data_dir)
    app.labor_database = database  # type: ignore[attr-defined]
    app.call_function_spec = function_spec  # type: ignore[attr-defined]

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/ask", methods=["POST"])
    def ask():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "").strip()

        if not question:
            return (
                jsonify(
                    {
                        "answer": "Please provide a question so I can search the knowledge base.",
                        "sources": [],
                    }
                ),
                400,
            )

        response = app.rag_pipeline.answer(question)  # type: ignore[attr-defined]
        return jsonify(response)

    @app.route("/schema", methods=["GET"])
    def schema():
        database = app.labor_database  # type: ignore[attr-defined]
        return jsonify(
            {
                "schema_text": database.get_schema_as_text(),
                "schema": database.get_schema_as_dict(),
                "function": app.call_function_spec,  # type: ignore[attr-defined]
            }
        )

    @app.route("/function/query", methods=["POST"])
    def function_query():
        payload = request.get_json(silent=True) or {}
        sql = payload.get("sql", "")
        max_rows = payload.get("max_rows")

        database = app.labor_database  # type: ignore[attr-defined]
        try:
            result = database.execute_query(sql, max_rows=max_rows)
        except (ValueError, sqlite3.Error) as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)