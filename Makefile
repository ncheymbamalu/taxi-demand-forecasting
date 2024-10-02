.PHONY: features training

# pushes the latest batch of validated and pre-processed NYC taxi demand data to Hopsworks
features:
	poetry run python src/pipelines/feature_pipeline.py

# evaluates the current model on the latest NYC taxi demand data and replaces it if necessary
training:
	poetry run python src/pipelines/training_pipeline.py
