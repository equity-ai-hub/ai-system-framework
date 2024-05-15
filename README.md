# Machine Learning Workflow


## Installing dependencies to work locally. 

Create and activate your Python environment (pyenv or conda). 
1. `pip install -e .`

### Docker 

Clone the repository. 
Before building the docker image, in the root directory: `mkdir logs` and `mkdir plugins`. Create these two folder, to locally store the airflow logs and the existent plugins.

```
docker build . --tag extending_airflow:latest
docker compose up airflow-init
docker compose up -d
```

Acess the airflow in `localhost:8080`
Username and password to log in the UI: `airflow`