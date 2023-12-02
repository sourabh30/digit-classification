docker push sourabhmlops.azurecr.io/dependency_digits
az acr build --image dependency_digits --registry sourabhmlops --file ./docker/DependencyDockerfile .

docker push sourabhmlops.azurecr.io/digits:v1
az acr build --image digits:v1 --registry sourabhmlops --file ./docker/Dockerfile .

