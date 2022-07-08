# Hyper Pramater Tuning for RelaySGD
lr_array=(0.01 0.025 0.05 0.075 0.1)
alpha_array=(1 0.1 0.01)
topo_array=("double_binary_trees" "chain" "binary_tree")

echo "Start hyper parameter grid search over lr=(0.01 - 0.1) for each alpha=(1, 0.1, 0.01)"
for topo in ${topo_array[@]}; do
    for alp in ${alpha_array[@]}; do
    echo "Summary relay ($topo) 8 GPUs alpha=$alp 50 epochs" > ./logs/summary_hpt_cifar10_relay-topo-$topo-a-$alp-8_gpus.txt
        for lr in ${lr_array[@]}; do
            echo "Executing with topo=$topo lr=$lr gpus=8 epochs=50 alpha=$alp"
            filename="./logs/hpt_cifar10_relay-topo-$topo-$lr_$alp-8_gpus.log"
            echo "Saving in $filename"
            python3 -m bagua.distributed.launch --nproc_per_node=8 benchmark.py --algorithm relay --topology $topo --lr $lr --epochs 50 --alpha $alp 2>&1 | tee $filename
            cur_acc=$(grep -oP "(\d+\/\d+)" $filename | tail -1 | grep -oP "(\d+)" | head -1)
            echo $cur_acc
            echo "lr=$lr acc=$cur_acc" >> ./logs/summary_hpt_cifar10_relay-topo-$topo-a-$alp-8_gpus.txt
        done
    done
done



