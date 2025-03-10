.PHONY: venv install check clean features train feature_pipeline training_pipeline

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
	dvc pull
	uv run python src/pipelines/feature.py

train:
	dvc pull
	uv run python src/pipelines/train.py

feature_pipeline:
	make features
	dvc add ./artifacts
	git add artifacts.dvc
	git commit -m "executing the feature pipeline"
	dvc push
	git push

training_pipeline:
	make train
	dvc add ./artifacts
	git add artifacts.dvc
	git commit -m "executing the training pipeline"
	dvc push
	git push
