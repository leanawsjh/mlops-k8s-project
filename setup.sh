eksctl create cluster \
  --name ml-cluster-jav \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --profile eks-user
