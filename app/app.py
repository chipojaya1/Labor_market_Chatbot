from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request

from .rag import RagPipeline

def create_app() -> Flask:
    app = Flask(__name__)

    corpus_path = os.getenv("RAG_CORPUS_PATH", "data/rag_corpus.json")
    app.rag_pipeline = RagPipeline(corpus_path=corpus_path)  # type: ignore[attr-defined]

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

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)