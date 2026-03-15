# AGENTS.md

This file provides guidelines for agentic coding agents working in this repository.

## Project Overview

This is a book reading assistant application with:
- **Backend**: Python FastAPI API (`src/api/`) with indexing/search logic (`src/index/`)
- **Frontend**: TypeScript React + Vite app (`src/client/`)

## Development Commands

### Python Backend

```bash
# Install dependencies (uses uv)
uv sync

# Run the API server (from project root)
uvicorn src.api.main:app --reload --port 8000

# Run a single test
pytest tests/test_util.py::test_create_batches

# Run all tests
pytest

# Format code
black src/
```

### Docker (use Makefile)

```bash
# Build images
make build

# Start services
make up

# Stop services
make down

# View logs
make logs
make logs-api
make logs-client
```

### TypeScript Frontend

```bash
# Navigate to client directory
cd src/client

# Install dependencies (uses bun)
bun install

# Run dev server
bun run dev

# Build for production
bun run build

# Lint
bun run lint

# Type check
bun run typecheck
```

## Code Style Guidelines

### Python

**Imports**
- Use absolute imports from `src` (e.g., `from src.index.book import Book`)
- Group imports: stdlib, third-party, local
- Sort within groups alphabetically

**Formatting**
- Line length: 88 characters (Black default)
- Use Black for formatting
- Type hints on function signatures

**Types**
- Use Pydantic `BaseModel` for data classes
- Prefer explicit type annotations
- Use `typing` module for generics (`List`, `Dict`, `Optional`, etc.)

**Naming**
- `snake_case` for functions, variables, file names
- `PascalCase` for classes, Pydantic models
- `SCREAMING_SNAKE_CASE` for constants

**Error Handling**
- Use `try/except` with specific exception types
- Log exceptions with `logging.exception()` before re-raising
- Raise `HTTPException` for API errors with appropriate status codes
- Validate path traversal in file operations

**Patterns**
- Use `@classmethod` for factory methods (e.g., `from_file`)
- Use `abc` module for abstract base classes
- Use dataclasses/Pydantic models for structured data
- Modularize API: use separate route files with their own `APIRouter`
- Keep shared config in `config.py`

### TypeScript/JavaScript

**Imports**
- Use path aliases (`@/` maps to `src/`)
- Group: external (react, radix), internal (@/), relative

**Formatting**
- ESLint handles formatting rules
- Use 2 spaces for indentation
- Use semicolons

**Types**
- Use explicit return types on functions
- Prefer interfaces for object shapes
- Use `type` for unions, primitives

**Naming**
- `camelCase` for variables, functions
- `PascalCase` for components, interfaces
- `kebab-case` for file names (components)

**Components**
- Use `.tsx` extension for React components
- Use functional components with hooks
- Export components as named exports
- Use React Query for async data fetching
- Use `sonner` for toasts

**Patterns**
- Use `cva` + `clsx` + `tailwind-merge` for component variants
- Follow Radix UI patterns for accessible components
- Use React Router for navigation

## API Structure

```
src/api/
├── __init__.py
├── config.py         # Paths, env vars (MONGO_URI, GUTENDEX_BASE_URL)
├── dependencies.py   # get_mongo_db, get_openrouter_client
├── models.py         # Pydantic request/response models
├── main.py           # App initialization, lifespan, router includes
└── routes/
    ├── search.py     # GET /books/search
    ├── books.py      # GET /books, POST /books/{id}/add, POST /books/request, PUT /books/{id}/progress
    └── reading.py    # GET /books/{id}/content, /toc, /summary, POST /ask
```

Each route module defines its own `APIRouter` with appropriate prefix (e.g., `/api/v1/books`).

## External APIs

### Gutendex API

Project Gutenberg's free ebook metadata API at `https://gutendex.com/`.

Key endpoints:
- `GET /books` - List books (supports `search`, `languages`, `topic`, `sort`, `page`, `mime_type` params)
- `GET /books/<id>` - Get single book

All searches use default params: `mime_type=text/plain` and `languages=en` to filter for plain text English books.

For random/popular results without a query, use jittered params (random page, topic, sort) to get varied results.

## Frontend Pages

```
src/client/src/pages/
├── HomePage.tsx    # Search/discover books, add to shelf, request books
├── Index.tsx       # Legacy redirect
└── NotFound.tsx
```

HomePage uses React Query with `enabled` flag to only fetch on user action (not on every keystroke).

## Key File Locations

- Backend entry: `src/api/main.py`
- Index logic: `src/index/`
- Frontend components: `src/client/src/components/`
- Frontend pages: `src/client/src/pages/`
- Types: `src/client/src/types/types.ts`
- API services: `src/client/src/services/bookService.ts`

## Testing

- Python tests: `tests/` directory
- Use pytest with `@pytest.mark.parametrize` for parametrized tests
- Run single test: `pytest path/to/test.py::test_name`
- Use `fastapi.testclient.TestClient` for endpoint testing

## Environment

- Python 3.10+
- Uses `python-dotenv` for env vars (`.env` file)
- API keys go in `.env` (already in `.gitignore`)
- MongoDB is optional for local development (app logs warning but continues)
- Docker Compose handles MongoDB for containerized development
