#!/bin/bash
echo "Starting MIA Backend on port ${PORT:-8000}"
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}