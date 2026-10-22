#!/bin/bash
# Starts the app on a Mac and prints the address to open on the phone (same Wi-Fi).
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q torch torchvision -r requirements.txt; }
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
