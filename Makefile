.PHONY: help build build-no-cache up down restart logs logs-api logs-client ps clean

build:
	docker compose build

build-no-cache:
	docker compose build --no-cache

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-client:
	docker compose logs -f client

ps:
	docker compose ps

clean:
	docker compose down -v
