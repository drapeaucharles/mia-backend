#!/bin/bash
# Start script for MIA Backend

# Run database migrations if needed
if [ -f "run_migrations.py" ]; then
    echo "Running database migrations..."
    python run_migrations.py
fi

# Start the FastAPI application
echo "Starting MIA Backend on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}