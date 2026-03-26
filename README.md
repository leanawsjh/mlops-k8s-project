# mlops-k8s-project
This is a test repo for AWS - k8

- use setup.sh file to set up kubernetes clusters.
- trouble shoot: if permession is denied --> chmod +x setup.sh
- run ./setup.sh

- to clean up the cluster --> chmod +x cleanup.sh
- run ./cleanup.sh

# How to run step by step
## 1. Train Models
- python models/train.py
- python models/generate_drift.py

## 2. Build & push Docker
- docker build -t your-dockerhub/ml-api:latest -f docker/Dockerfile .
- docker push your-dockerhub/ml-api:latest

## Deploy on kubernetes
- kubectl apply -f k8s/deployment-v1.yaml
- kubectl apply -f k8s/service.yaml
