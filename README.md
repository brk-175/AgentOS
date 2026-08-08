# AgentOS

AI-powered GitHub code-fix assistant for engineering teams.

Connect a GitHub repository, select an issue or PR, and AgentOS will investigate
the problem, retrieve relevant repo context, generate a fix, create a
branch, commit the changes, and open a Pull Request for human review.

## Architecture

```
Browser (Next.js) → FastAPI → Celery worker → LangGraph agent
                          ├── MCP Client → GitHub MCP server (stdio) → GitHub REST
                          ├── RAG retriever → pgvector (PostgreSQL)
                          ├── LLM → OpenRouter (agent + embeddings + judge)
                          └── SSE (Redis pub/sub) → live UI trace
```

## Stack

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS v4, shadcn/ui
- **Backend:** FastAPI (Python 3.12, async), SQLAlchemy, Alembic
- **Agent:** LangGraph (single agent, staged workflow)
- **LLM:** OpenRouter (agent + embeddings + eval judge)
- **Tools:** Custom GitHub MCP server (MCP Python SDK v2)
- **Data:** PostgreSQL 16 + pgvector, Redis (cache/queue)
- **Workers:** Celery + Redis
- **Ops:** Docker, Kubernetes, OpenTelemetry, Prometheus/Grafana, GitHub Actions

## Repository layout

```
backend/            FastAPI application (api, services, agent, rag, mcp client)
github-mcp-server/  Standalone GitHub MCP server (stdio)
frontend/           Next.js application
eval/               Golden dataset + evaluation reports
deploy/             Docker + Kubernetes manifests
docs/               Architecture and runbooks
```

## Development

Prerequisites: Docker, Docker Compose, Python 3.12+, Node 22+.

See `docs/` for the full runbook. Quick start:

```bash
cp .env.example .env        # fill in secrets (see .env.example)
make dev                    # docker compose up
make install                # Poetry: backend deps + dev tools
make migrate                # run DB migrations
```

## Contributing

See `CONTRIBUTING.md` in `docs/`.

> **Note:** MVP scope (v1): single agent, single investigate → fix → PR
> workflow. No multi-agent, advanced RAG, or SaaS billing in v1.