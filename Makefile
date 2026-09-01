.PHONY: help ui backend test

UI_PORT ?= 4173
API_HOST ?= 0.0.0.0
API_PORT ?= 8080

help:
	@echo "Toronto demo commands:"
	@echo "  make ui       Start the slideshow frontend on port $(UI_PORT)"
	@echo "  make backend  Start the control API on $(API_HOST):$(API_PORT)"
	@echo "  make test     Run the Python test suite"

ui:
	uv run python -m http.server $(UI_PORT) --directory ui

backend:
	uv run uvicorn api.app:app --host $(API_HOST) --port $(API_PORT)

test:
	uv run pytest -q
