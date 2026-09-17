.PHONY: setup validate run

setup:
	python -m pip install -r requirements.txt

validate:
	python scripts/validate_repository.py
	python -m py_compile src/generate_paper_results.py

run:
	python src/generate_paper_results.py --data-dir data/raw --output-dir results/generated

