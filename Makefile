PYTHON=.venv/bin/python3
.PHONY: help test

all: run

init:
	python3.9 -m virtualenv .venv

run: ## run the pipeline (train)
	$(PYTHON) src/train.py \
		debug=false

debug: ## run the pipeline (train) with debugging enabled
	$(PYTHON) src/train.py \
		debug=true

data: ## download the mnist data
	wget https://pjreddie.com/media/files/mnist_train.csv -O data/mnist_train.csv
	wget https://pjreddie.com/media/files/mnist_test.csv -O data/mnist_test.csv
test:
	find . -iname "*.py" | entr -c pytest

install:
	$(PYTHON) -m pip install -r requirements.txt

help: ## display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:
	conda env updates -n ${CONDA_ENV} --file environment.yml
