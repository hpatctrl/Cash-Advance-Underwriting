.PHONY: all test notebook
all:
	python run_pipeline.py
test:
	pytest -q
notebook:
	python notebooks/build_notebook.py
