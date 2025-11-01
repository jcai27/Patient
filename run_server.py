#!/usr/bin/env python3
"""Run the persona chatbot API server."""
import uvicorn
from src.server.api import app

if __name__ == "__main__":
    uvicorn.run("src.server.api:app", host="0.0.0.0", port=8000, reload=True)

