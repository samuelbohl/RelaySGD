lr_array=(0.001 0.005 0.01 0.05 0.1 1)
mom_array=(0.5 0.9 0.99)

for lr in ${lr_array[@]}; do
  for mom in ${mom_array[@]}; do
    echo "Executing with algo=relay lr=$lr momentum=$mom"
    python3 -m bagua.distributed.launch --nproc_per_node=8 benchmark.py --algorithm relay --epochs 25 2>&1 | tee ./logs/gs_cifar10_$lr-$mom.log
  done
done
