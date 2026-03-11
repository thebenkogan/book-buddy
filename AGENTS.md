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

# Type check
mypy src/

# Lint (if using ruff)
ruff check src/
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

**Patterns**
- Use `sonner` for toasts
- Use `cva` + `clsx` + `tailwind-merge` for component variants
- Follow Radix UI patterns for accessible components
- Use React Router for navigation

## Key File Locations

- Backend entry: `src/api/main.py`
- Index logic: `src/index/`
- Frontend components: `src/client/src/components/`
- Frontend pages: `src/client/src/pages/`
- Types: `src/client/src/types/types.ts`
- API services: `src/client/src/services/`

## Testing

- Python tests: `tests/` directory
- Use pytest with `@pytest.mark.parametrize` for parametrized tests
- Run single test: `pytest path/to/test.py::test_name`

## Environment

- Python 3.10+
- Uses `python-dotenv` for env vars (`.env` file)
- API keys go in `.env` (already in `.gitignore`)