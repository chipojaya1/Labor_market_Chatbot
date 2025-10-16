# Data Science Career Copilot

A lightweight Retrieval-Augmented Generation (RAG) prototype that provides career guidance for aspiring data scientists using curated 2024 labor market data.  The chatbot combines structured salary statistics from BLS and Glassdoor with skill signals mined from a job postings corpus.

## Features

- **Job Posting Skill Insights**: Skill-focused RAG corpus built from the 2024 job postings dataset
- **Dual Data Sources**: JSON corpus or SQLite database with FTS indexing
- **Transparent Citations**: All responses include explicit source references
- **RESTful API**: `/ask` endpoint for programmatic access
- **Web Interface**: Minimal chat UI for interactive queries
- **Cross-Platform**: Supports WSL/Linux, Windows PowerShell, and macOS

## Quick Start

### Basic Setup (JSON Corpus)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Build the job postings RAG corpus (writes data/rag_corpus.json)
python scripts/build_jobpostings_corpus.py

export FLASK_APP=app.app
flask run --host=127.0.0.1
```

Visit `http://127.0.0.1:5000` to access the chat interface.

### Database Setup (Recommended)
For better performance with larger datasets:

```bash
# Install additional dependencies
pip install pandas scikit-learn

# Generate the JSON corpus (if not already built)
python scripts/build_jobpostings_corpus.py

# Build the database (filters the CSVs to 2024 where applicable)
python scripts/create_db.py

# Run with database backend
export RAG_DB_PATH=data/labor_market.db
export FLASK_APP=app.app
flask run --host=127.0.0.1 --port=5001
```

## API Usage

### Ask a Question
```bash
POST /ask
Content-Type: application/json

{
  "question": "What skills are needed for Data Scientists?",
  "max_sources": 3
}
```

### Response Format
```json
{
  "answer": "Synthesized answer with bullet points...",
  "sources": [
    {
      "id": "doc_123",
      "title": "Document Title",
      "source_name": "BLS OEWS",
      "source_url": "https://...",
      "last_updated": "2024-01-01",
      "score": 0.856
    }
  ]
}
```

## Project Structure

```
├── app/
│   ├── app.py              # Flask application factory
│   ├── rag.py              # RAG pipeline with TF-IDF retrieval
│   └── static/             # Web interface assets
├── data/
│   ├── rag_corpus.json     # Default document corpus
│   └── labor_market.db     # Generated SQLite database
├── scripts/
│   ├── create_db.py        # Database creation utility
│   ├── build_jobpostings_corpus.py  # Generate skill-focused RAG corpus
│   └── supervise_flask.py  # Process supervisor
└── requirements.txt        # Python dependencies
```

## Core Components

### RAG Pipeline (`rag.py`)
- TF-IDF vectorization with cosine similarity
- Optional SQLite FTS5 integration for faster retrieval
- Support for both JSON and database backends
- Automatic source citation and table row sampling

### Database Layer (`database.py`)
- In-memory SQLite with CSV dataset loading
- Schema introspection and safe query execution
- Column name normalization for SQL compatibility

### Flask Application (`app.py`)
- Environment-based configuration (JSON vs DB corpus)
- RESTful API with error handling
- Template rendering for web interface

## Configuration

### Environment Variables
- `RAG_DB_PATH`: Path to SQLite database (optional)
- `RAG_CORPUS_PATH`: Path to JSON corpus (default: `data/rag_corpus.json`)
- `FLASK_APP`: Application entry point (`app.app`)
- `PORT`: Server port (default: 5000)

### Database Tables
The structured SQL layer uses three main data sources:
- `glassdoor_salary`: Salary insights from Glassdoor (2024 entries only)
- `bls_macro_indicators`: BLS economic indicators (filtered to 2024)
- `oews_salary`: Occupational Employment and Wage Statistics by state (2024)

Skill-centric answers are powered by the `data/rag_corpus.json` corpus, generated from the 2024 job postings dataset via `scripts/build_jobpostings_corpus.py`.

## Advanced Features

### Supervisor Mode
For production-like reliability:
```bash
python scripts/supervise_flask.py --port 5001
```

### Corpus Expansion
Generate additional documents from CSV data:
```bash
python scripts/generate_rag_from_csv.py
```

## Development

### Testing
```bash
# Test the RAG pipeline
python scripts/test_rag_db.py

# Demo API client
python scripts/demo_query.py
```

### Dependencies
- **Flask**: Web framework
- **scikit-learn**: TF-IDF vectorization and similarity
- **pandas**: Data processing for database creation
- **SQLite**: Embedded database engine

## Troubleshooting

**Common Issues:**
- Ensure virtual environment is activated
- Check data files exist in `data/` directory
- Verify all dependencies are installed
- Use `--host=127.0.0.1` for cross-platform compatibility

**Database Errors:**
- Run `python scripts/create_db.py` to regenerate database
- Check file permissions in `data/` directory
- Ensure pandas and scikit-learn are installed for DB features

The application automatically falls back to JSON corpus if database is unavailable, ensuring graceful degradation.
