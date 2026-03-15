import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from openrouter import OpenRouter

from src.api import config
from src.api.routes import books, reading, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    app.state.openrouter_client = client

    try:
        mongo_client = AsyncIOMotorClient(config.MONGO_URI)
        app.state.mongo_client = mongo_client
        app.state.mongo_db = mongo_client[config.MONGO_DB_NAME]

        db = app.state.mongo_db
        await db["books"].create_index("book_id", unique=True)
        await db["reading_progress"].create_index(
            [("user_id", 1), ("book_id", 1)], unique=True
        )
    except Exception as e:
        logging.warning(f"MongoDB not available: {e}")
        app.state.mongo_client = None
        app.state.mongo_db = None

    try:
        yield
    finally:
        if app.state.mongo_client:
            app.state.mongo_client.close()


app = FastAPI(title="Book API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(books.router)
app.include_router(reading.router)
app.include_router(search.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
