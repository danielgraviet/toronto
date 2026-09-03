.PHONY: help ui talk backend test demo demo-vllm smoke-remote smoke-remote-vllm

UI_PORT ?= 4173
TALK_PORT ?= 4174
API_HOST ?= 0.0.0.0
API_PORT ?= 8080

help:
	@echo "Toronto demo commands:"
	@echo "  make ui       Start the interactive stage UI on port $(UI_PORT)"
	@echo "  make talk     Start the talk-day slide deck on port $(TALK_PORT)"
	@echo "  make backend  Start the control API on $(API_HOST):$(API_PORT)"
	@echo "  make demo     Run the interactive GRPO CLI demo (HF rollouts)"
	@echo "  make demo-vllm  Same demo with vLLM colocated GRPO rollouts"
	@echo "  make test     Run the Python test suite"
	@echo "  make smoke-remote      Remote GPU preflight + 1-step HF smoke (needs DAYTONA_API_KEY)"
	@echo "  make smoke-remote-vllm Same with vLLM backend (catches CUDA/vLLM import issues)"

ui:
	uv run python -m http.server $(UI_PORT) --directory ui

talk:
	uv run python -m http.server $(TALK_PORT) --directory talk

backend:
	uv run uvicorn api.app:app --host $(API_HOST) --port $(API_PORT)

test:
	uv run pytest -q

demo:
	uv run python -m demo

demo-vllm:
	TORONTO_GENERATION_BACKEND=vllm uv run python -m demo

smoke-remote:
	uv run python -m runners.gpu --remote --real-grpo-smoke \
		--profile stage --gpu-type RTX-PRO-6000 --task-id two_sum_plus \
		--train-steps 1 --pool-size 4 --eval-samples 4

smoke-remote-vllm:
	TORONTO_GENERATION_BACKEND=vllm uv run python -m runners.gpu --remote --real-grpo-smoke \
		--profile stage --gpu-type RTX-PRO-6000 --task-id two_sum_plus \
		--train-steps 1 --pool-size 4 --eval-samples 4 --generation-backend vllm
