.PHONY: install lint test serve gradio eval build-index docker-build

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	black src/ tests/ && isort src/ tests/ --profile black && flake8 src/ tests/ && bandit -r src/ -ll -ii

test:
	python -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

serve:
	python -m uvicorn src.api.app:app --reload --port 8000

gradio:
	python -m src.api.gradio_demo

eval:
	python -m src.evaluation.evaluate

build-index:
	python -m src.data.dataset && python -m src.data.chunker && python -m src.retrieval.index

docker-build:
	docker build -t b4-semantic-search-faiss .
