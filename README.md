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

# Demo
move `model_data` directory to the root of repo

move `data` directory to the root of repo

sudo docker build -t prostate_gpu -f Dockerfile-demo .

sudo docker run -p 9000:8888  -it -v /home/server/prostate_service/data:/app/data:rw  prostate_gpu

jupyter notebook --ip=0.0.0.0 --allow-root

open localhost:9000







