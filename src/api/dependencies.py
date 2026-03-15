from typing import Optional

from fastapi import Request

from motor.motor_asyncio import AsyncIOMotorDatabase
from openrouter import OpenRouter


def get_openrouter_client(request: Request) -> OpenRouter:
    return request.app.state.openrouter_client


def get_mongo_db(request: Request) -> Optional[AsyncIOMotorDatabase]:
    return request.app.state.mongo_db
