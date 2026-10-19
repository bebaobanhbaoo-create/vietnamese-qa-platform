.PHONY: install scrape process index eval query test lint format typecheck check

install:
	pip install -e ".[dev]"
	pre-commit install

scrape:
	huflit-qa scrape

process:
	huflit-qa process

index:
	huflit-qa index

eval:
	huflit-qa eval --output eval/results/latest.json

query:
	huflit-qa query "$(Q)"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

check: lint typecheck test
