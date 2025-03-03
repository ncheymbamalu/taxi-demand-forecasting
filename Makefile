.PHONY: venv install check clean

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
	rm -rf `find . -type d -name catboost_info`
	rm -rf .ruff_cache
	rm -rf logs
