.PHONY: features training web_app

# generates a new batch of hourly taxi rides and pushes it to the Hopsworks feature store
features:
	poetry run python src/pipelines/feature_pipeline.py

# trains and evaluates a new model and pushes it to the Hopswork model registry
training:
	poetry run python src/pipelines/training_pipeline.py

# starts the Streamlit web application
web_app:
	poetry run streamlit run src/app.py