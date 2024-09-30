.PHONY: features

# pushes the latest batch of validated and pre-processed NYC taxi demand data to Hopsworks
features:
	poetry run python src/pipelines/feature_pipeline.py
