SHELL := /bin/bash

source:
	source .venv/bin/activate && echo "Virtual environment activated" && which python

install:
	uv sync

run:
	uv run python main.py

ga:
	@if [ -z "$(m)" ]; then \
		echo "Usage: make ga MSG='your commit message'"; \
		exit 1; \
	fi
	git add .
	git commit -m "$(m)"
	git push
