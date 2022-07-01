# first arg: algorithm (realy, allreduce, decentralized)
# second arg: number of gpus
# third arg: number of epochs
lr_array=(0.001 0.005 0.01 0.05 0.1 0.5)
alpha_array=(1 0.1 0.01)
cur_best=0

echo "Start hyper parameter grid search over lr=(0.001 - 0.5) for each alpha=(1, 0.1, 0.01)"
for alp in ${alpha_array[@]}; do
  echo "Summary $1 $2 alpha=$alp GPUs $3 epochs" > ./logs/summary_hp_cifar10_a-$alp-$2_gpus.txt
  for lr in ${lr_array[@]}; do
    echo "Executing with algo=$1 lr=$lr gpus=$2 epochs=$3 alpha=$alp"
    filename="./logs/hp_cifar10_$lr_$alp-$2_gpus.log"
    echo "Saving in $filename"
    python3 -m bagua.distributed.launch --nproc_per_node=$2 benchmark.py --algorithm $1 --epochs $3 --alpha $alp 2>&1 | tee $filename
    cur_acc=$(grep -oP "(\d+\/\d+)" $filename | tail -1 | grep -oP "(\d+)" | head -1)
    echo $cur_acc
    if (( $cur_acc > $cur_best )); then
      cur_best=$cur_acc
      best_lr=$lr
    fi
    echo "lr=$lr acc=$cur_acc" >> ./logs/summary_hp_cifar10_a-$alp-$2_gpus.txt
  done
  echo "Best Acc for $1 with alpha=$alp on $2 gpus: acc=$cur_best lr=$best_lr" > ./logs/best_hp_cifar10_a-$alp-$2_gpus.txt
done


