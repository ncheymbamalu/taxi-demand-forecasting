.PHONY: venv install check clean features train

venv: pyproject.toml
	uv venv
	. .venv/bin/activate

install: .venv
	uv sync

check: install
	uv tool run isort src
	uv tool run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache
	rm -rf logs

features:
	uv run python src/pipelines/feature.py

train:
	uv run python src/pipelines/train.py
