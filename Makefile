# Commands:
#   make fix        # Format code and auto-fix lint with Black + Ruff
#   make lint       # Run Ruff linter
#   make test       # Run tests with pytest
#   make check      # Run lint, typecheck, and test
#   make ci         # Run fix, lint, typecheck, and test (CI pipeline)
#   make coverage   # Generate test coverage report (HTML)
#   make clean      # Remove temp/cache files

.PHONY: fix lint test check ci coverage clean

fix:
	black .
	ruff check --fix .

lint:
	ruff check .

test:
	pytest

check: lint test

ci: fix lint test

coverage:
	pytest --cov=src --cov-report=html

clean:
	@echo Cleaning up test artifacts and caches...
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .coverage del /q .coverage
	@if exist htmlcov rmdir /s /q htmlcov
	@powershell -Command "Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"

