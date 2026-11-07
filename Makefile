.PHONY: test lint lint-fix
test:
	python -m pytest -q
lint:
	ruff check .
	ruff format --check .
	mypy src tests
lint-fix:
	ruff check --fix .
	ruff format .
