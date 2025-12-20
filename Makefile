# Makefile for ResearchAgent project
# Provides common commands for development and deployment

.PHONY: help install install-dev test lint format type-check clean run-pipeline run-streamlit build docs

# Default target
help:
	@echo "ResearchAgent - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run tests with coverage"
	@echo "  make lint             Run linting (ruff)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make type-check       Run type checking (mypy)"
	@echo "  make clean            Clean build artifacts and cache"
	@echo ""
	@echo "Running:"
	@echo "  make run-pipeline     Run full data pipeline"
	@echo "  make run-streamlit    Start Streamlit web interface"
	@echo ""
	@echo "Distribution:"
	@echo "  make build            Build package distribution"
	@echo "  make docs             Generate documentation"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x

# Code quality
lint:
	ruff check src/ tests/ streamlit_app.py

lint-fix:
	ruff check src/ tests/ streamlit_app.py --fix

format:
	black src/ tests/ streamlit_app.py
	isort src/ tests/ streamlit_app.py

format-check:
	black --check src/ tests/ streamlit_app.py
	isort --check src/ tests/ streamlit_app.py

type-check:
	mypy src/ --install-types --non-interactive

# Security
security:
	bandit -r src/ -f screen
	safety check

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

# Running the application
run-pipeline:
	python src/main.py

run-streamlit:
	streamlit run streamlit_app.py

# Building
build: clean
	python -m build

# Documentation
docs:
	@echo "Documentation generation not yet configured"
	@echo "Consider using Sphinx: pip install sphinx sphinx-rtd-theme"

# All quality checks
check-all: format-check lint type-check test

# Pre-commit hook setup
setup-pre-commit:
	pre-commit install
