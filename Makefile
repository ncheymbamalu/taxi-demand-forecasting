.PHONY: venv install check clean pull features push_features feature_pipeline

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

pull:
	dvc pull

features: pull
	uv run python src/pipelines/feature.py

push_features:
	dvc add ./artifacts
	git add artifacts.dvc
	git commit -m "executing the feature pipeline"; dvc push
	git push
	rm -rf artifacts

feature_pipeline: features push_features clean
