# Machine Learning Workflow


## Installing dependencies to work locally. 

Create and activate your Python environment (pyenv or conda). 
1. `pip install -e .`

### Docker 

Clone the repository.
```
docker build . --tag extending_airflow:latest
docker compose up airflow-init
docker compose up -d
```