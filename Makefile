.PHONY: features

# generates a new batch of hourly taxi rides and pushes it to the Hopsworks feature store
features:
	poetry run python src/pipelines/feature_pipeline.py