mutants:
	mutmut run \
		--paths-to-mutate HCLcat

.PHONY: clean install mutants tests

install:
	pip install --editable .

format:
	black -l 100 HCLcat

tests: install
	pytest --verbose

clean:
	rm --recursive --force test/__pycache__
	rm --recursive --force HCLcat/__pycache__
	rm --recursive --force HCLcat.egg-info
	rm --force .mutmut-cache
