# RUN experiment 1 with all the previousliy tuned learning rates in `lr_array`
algo_array=("relay" "relay" "relay" "relay" "relay" "relay" "relay" "relay" "relay" "decentralized" "decentralized" "decentralized" "allreduce" "allreduce" "allreduce")
topo_array=("chain" "chain" "chain" "binary_tree" "binary_tree" "binary_tree" "double_binary_trees" "double_binary_trees" "double_binary_trees" "-" "-" "-" "-" "-" "-")
lr_array=(0.1 0.075 0.1 0.075 0.075 0.1 0.1 0.1 0.1 0.025 0.05 0.075 0.1 0.075 0.05)
alpha_array=(0.01 0.1 1 0.01 0.1 1 0.01 0.1 1 0.01 0.1 1 0.01 0.1 1 0.01 0.1 1 0.01 0.1 1 0.01 0.1 1)


for (( i=0; i<${#lr_array[@]}; i++ ));
do
    echo "Running with algo=${algo_array[$i]} topo=${topo_array[$i]} lr=${lr_array[$i]} alpha=${alpha_array[$i]}"
    filename="./logs/exp1_cifar10_${algo_array[$i]}_t-${topo_array[$i]}_lr-${lr_array[$i]}_a-${alpha_array[$i]}.log"
    echo "Saving in $filename"
    python3 -m bagua.distributed.launch --nproc_per_node=8 benchmark.py --algorithm ${algo_array[$i]} --topology ${topo_array[$i]} --lr ${lr_array[$i]} --alpha ${alpha_array[$i]} 2>&1 | tee $filename
done
