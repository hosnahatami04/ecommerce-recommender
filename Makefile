.PHONY: install download verify test lint format

install:
	pip install -r requirements.txt

download:
	python -m src.data.download

verify:
	python -m src.data.download --verify-only

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
