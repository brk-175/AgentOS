# AgentOS development Makefile (portable: uses Python where possible)

SHELL := /bin/bash
COMPOSE ?= docker compose
POETRY ?= poetry

.PHONY: help dev down ps logs backend-dev migrate revision install shell lint format typecheck test test-eval ci

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

## --- Local development ---

dev: ## Start full local stack (backend + worker + frontend + db + redis)
	$(COMPOSE) up --build

down: ## Stop and remove local stack
	$(COMPOSE) down

ps: ## Show running containers
	$(COMPOSE) ps

logs: ## Tail logs from all services
	$(COMPOSE) logs -f --tail=200

## --- Backend (Poetry) ---

backend-dev: ## Run backend with hot reload on port 8000
	cd backend && $(POETRY) run uvicorn agentos.app:app --host 0.0.0.0 --port 8000

install: ## Install backend deps + dev group via Poetry
	cd backend && $(POETRY) install

shell: ## Open a shell inside the Poetry-managed venv
	cd backend && $(POETRY) shell

migrate: ## Rebuild local db from migrations
	cd backend && $(POETRY) run alembic upgrade head

revision: ## Auto-generate a migration after model changes (usage: make revision msg="desc")
	cd backend && $(POETRY) run alembic revision --autogenerate -m "$(msg)"

## --- Frontend ---

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

## --- Quality ---

lint: ## Run linters
	cd backend && $(POETRY) run ruff check src tests
	cd backend && $(POETRY) run mypy src

format: ## Auto-format code
	cd backend && $(POETRY) run ruff format src tests

typecheck: ## Frontend type-check + backend mypy
	cd frontend && npx tsc --noEmit

test: ## Run backend unit tests
	cd backend && $(POETRY) run pytest -m "not eval"

test-eval: ## Run evaluation regression suite
	cd backend && $(POETRY) run pytest -m eval

ci: lint typecheck test test-eval ## Run all quality gates