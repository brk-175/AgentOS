# AgentOS development Makefile (portable: uses Python where possible)

SHELL := /bin/bash
PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: help dev down ps logs migrate seed lint typecheck test eval ci

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

## --- Backend ---

# Shell into backend python env usage: `make backend-shell`
backend-env: ## Create/enter backend virtualenv
	cd backend && $(PYTHON) -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"

migrate: ## Create Venv (re)build local db from migrations
	cd backend && alembic upgrade head

revision: ## Auto-generate a migration after model changes (usage: make revision msg="desc")
	cd backend && alembic revision --autogenerate -m "$(msg)"

## --- Frontend ---

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

## --- Quality ---

lint: ## Run linters
	cd backend && ruff check src tests
	cd backend && mypy src

format: ## Auto-format code
	cd backend && ruff format src tests

typecheck: ## Frontend type-check + backend mypy
	cd frontend && npx tsc --noEmit

test: ## Run backend unit tests
	cd backend && pytest -m "not eval"

test-eval: ## Run evaluation regression suite
	cd backend && pytest -m eval

ci: lint typecheck test test-eval ## Run all quality gates