.PHONY: install test lint clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q tests/ || python -m pytest -q .

lint:
	python -m py_compile $$(find . -name '*.py' ! -path '*/.venv/*')

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
