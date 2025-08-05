# MIA Backend

A FastAPI backend for MIA (Decentralized AI Customer Support Assistant), where miners run Mixtral to process chat jobs.

## Features

- **Job Queue System**: Redis-based job queue for distributing chat requests to miners
- **RESTful API**: FastAPI endpoints for chat, job management, and miner registration
- **PostgreSQL Database**: Stores businesses, chat logs, and miner information
- **Docker Support**: Ready for Railway deployment
- **Health Checks**: Built-in health monitoring endpoints

## API Endpoints

### Core Endpoints

- `POST /chat` - Submit a new chat message and create a job
- `GET /job/next` - Miners fetch the next available job
- `POST /job/result` - Miners submit job results
- `POST /register_miner` - Register a new miner

### Utility Endpoints

- `GET /` - Basic health check
- `GET /health` - Detailed health check with service status

## Project Structure

```
mia-backend/
├── main.py           # FastAPI application and routes
├── db.py            # SQLAlchemy models and database setup
├── queue.py         # Redis queue implementation
├── schemas.py       # Pydantic models for request/response
├── utils.py         # Helper functions
├── Dockerfile       # Docker configuration for Railway
├── requirements.txt # Python dependencies
├── railway.json     # Railway deployment configuration
├── .env.example     # Environment variables template
├── .gitignore      # Git ignore patterns
└── README.md       # This file
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- Docker (for containerized deployment)

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd mia-backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start PostgreSQL and Redis:
```bash
# Using Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres
docker run -d -p 6379:6379 redis
```

6. Run the application:
```bash
python main.py
# Or use uvicorn directly
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Deployment

Build and run with Docker:

```bash
docker build -t mia-backend .
docker run -p 8000:8000 --env-file .env mia-backend
```

## Railway Deployment

1. Create a new Railway project
2. Add PostgreSQL and Redis services
3. Connect your GitHub repository
4. Railway will automatically detect the Dockerfile and deploy

Environment variables required in Railway:
- `DATABASE_URL` - PostgreSQL connection string (provided by Railway)
- `REDIS_URL` - Redis connection string (provided by Railway)

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost/db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `PORT` | Server port (optional) | `8000` |

## Database Models

### Business
- `id`: Primary key
- `name`: Business name
- `contact_email`: Contact email
- `contact_phone`: Contact phone
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### ChatLog
- `id`: Primary key
- `session_id`: Chat session identifier
- `message`: Message content
- `role`: Message role (user/assistant)
- `timestamp`: Message timestamp
- `business_id`: Associated business (optional)

### Miner
- `id`: Primary key
- `name`: Miner name
- `auth_key`: Authentication key
- `job_count`: Number of jobs processed
- `created_at`: Registration timestamp
- `last_active`: Last activity timestamp

## Job Queue Structure

Jobs are stored in Redis with the following structure:
```json
{
  "job_id": "uuid",
  "prompt": "user message",
  "context": "optional context",
  "session_id": "uuid",
  "business_id": 1,
  "timestamp": "2024-01-01T00:00:00"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license here]