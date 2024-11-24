.PHONY: install check features train frontend clean runner_features runner_train runner_frontend

install: pyproject.toml
	poetry install

check: install
	poetry run flake8 src

features:
	poetry run python src/pipelines/feature_pipeline.py

train:
	poetry run python src/pipelines/training_pipeline.py

frontend:
	poetry run streamlit run src/app.py

clean:
	rm -rf `find . -type d -name __pycache__`

runner_features: check features clean

runner_train: check train clean

runner_frontend: check frontend clean
