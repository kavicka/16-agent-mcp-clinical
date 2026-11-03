.PHONY: sync tasks eval plot mcp demo test lint

sync:
	uv sync

tasks:            ## rebuild eval/tasks.jsonl and record the API cache (needs internet)
	CTGOV_CACHE=record uv run python eval/build_tasks.py

eval:             ## needs Ollama with $(LLM_MODEL) pulled (or LLM_PROVIDER=transformers)
	CTGOV_CACHE=record uv run python eval/run_eval.py   # new queries are fetched once and cached

plot:
	uv run python plot.py

mcp:              ## the MCP server on stdio (normally started by an MCP client, not by hand)
	uv run python -m mcp_server.server

demo:             ## one task with a printed trace
	uv run python demo.py "Which of the trials NCT06750289 and NCT01842607 enrolled more participants, and by how many?"

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
