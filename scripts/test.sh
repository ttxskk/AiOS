srun \
-p a2584704-17c1-47e9-aeb5-bf500faf7d0f \
--workspace-id  fe790a91-4552-4cfc-bffe-8d7a657aee7b \
--priority normal \
--container-image registry.st-sh-01.sensecore.cn/zoetrope/aios:20240119-00h00m27s \
--container-mounts 96e0b441-aaf6-11ee-a837-ee00953f5cc3:/mnt/AFS_sunqingping,f7c98e03-8dc9-11ee-89cf-a6e6d39bfd56:/mnt/AFS_Zoetrope,7bdc0e33-a3ca-11ee-9988-664fb1f07227:/mnt/AFS_datasets \
-j npz3 \
-r N1lS.Ia.I20.8 \
-N 1 \
--framework pytorch \
-d StandAlone \
sleep 1d

# -d AllReduce \
# -d StandAlone \