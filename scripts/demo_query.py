"""Demo script: POST a question to the running Flask /ask endpoint and print the JSON response."""
from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:5001/ask"
PAYLOAD = {"question": "What are the top skills required for data scientists in 2024?"}

req = Request(URL, data=json.dumps(PAYLOAD).encode("utf-8"), headers={"Content-Type": "application/json"})
with urlopen(req, timeout=10) as resp:
    body = resp.read().decode("utf-8")
    try:
        obj = json.loads(body)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(body)
