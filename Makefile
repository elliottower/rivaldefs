.PHONY: test lint typecheck build all

test:
	uv run --no-project --with numpy --with pytest --with-editable . python -m pytest tests/ -q

lint:
	uv run --no-project --with ruff ruff check src tests

typecheck:
	uv run --no-project --with mypy --with numpy mypy src

build:
	uv build

all: test lint typecheck
