# How to use this Makefile:
# 
# Common commands:
#   make format     # Format code using Black
#   make lint       # Lint code using Flake8
#   make test       # Run tests with pytest
#   make coverage   # Generate test coverage report (HTML)
#   make clean      # Remove temporary and cache files
#   make check      # Run format, lint, and test in sequence
#
# Usage:
#   Run `make <target>` in your terminal, e.g.:
#     make test



.PHONY: format lint test coverage clean

format:
	black .

lint:
	flake8 .

test:
	pytest

coverage:
	pytest --cov-report=html

clean:
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

check: format lint test 