.PHONY: setup install lint test train evaluate serve deploy monitor clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pre-commit install
	dvc init
	@echo "Setup complete. Activate venv: source .venv/bin/activate"

install:
	pip install -r requirements.txt

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	black src/ tests/

test-unit:
	pytest tests/unit/ -m unit -v

test-integration:
	pytest tests/integration/ -m integration -v

test-model:
	pytest tests/model/ -m model -v

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

ingest:
	python -m src.data.ingest

validate:
	python -m src.data.validate

preprocess:
	python -m src.data.preprocess

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

pipeline: ingest validate preprocess train evaluate
	@echo "Full pipeline complete."

serve-local:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t mlops-fraud-detection:latest .

docker-run:
	docker run -p 8000:8000 mlops-fraud-detection:latest

ecr-push:
	./scripts/deploy.sh push

deploy-lambda:
	./scripts/deploy.sh deploy

deploy: docker-build ecr-push deploy-lambda

monitor-drift:
	python -m src.monitoring.drift_detection

monitor-performance:
	python -m src.monitoring.performance

dvc-push:
	dvc push

dvc-pull:
	dvc pull

mlflow-ui:
	mlflow ui --port 5000

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf mlruns/ mlartifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
