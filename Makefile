.DEFAULT_GOAL := help

#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y beyond_the_board || :
	@pip install -e .

install:
	@pip install -e .

requirements:
	@pip install -r requirements.txt


#################### API ACTIONS ###################

run_api:
	uvicorn beyond_the_board.api.chess:app --reload --host 0.0.0.0 --port 8000


##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr beyond_the_board.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -fr **/.ipynb_checkpoints
	@rm -fr .pytest_cache


##################### HELP #####################

help:
	@echo "Available commands:"
	@echo "  make install              - Install package in editable mode"
	@echo "  make requirements         - Install dependencies from requirements.txt"
	@echo "  make reinstall_package    - Uninstall and reinstall package"
	@echo "  make run_api              - Run FastAPI server for chess evaluation"
	@echo "  make clean                - Remove Python cache files and build artifacts"

##################### DOCKER #####################
docker_build:
	@docker build --tag=beyond_the_board .

docker_run:
	@docker run -it -e PORT=8000 -p 8000:8000 --env-file .env \
	-v ~/code/CousinCombey/le-wagon-bootcamp-475009-c2d7ce68ab5a.json:/app/gcp-key.json \
	-e GOOGLE_APPLICATION_CREDENTIALS=app/gcp-key.json \
	beyond_the_board

docker_running:
	@docker ps
