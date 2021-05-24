minikube start
eval $(minikube docker-env)
docker build -t prostate-api -f Dockerfile-api .
docker build -t prostate-gpu -f Dockerfile-gpu .



kubectl apply -f api-configmap.yaml
kubectl apply -f api.yaml
kubectl apply -f gpu.yaml

minikube service api-service


kubectl exec -it gpu-8f558bd9d-j2nnb -- /bin/bash
kubectl delete deployment gpu
