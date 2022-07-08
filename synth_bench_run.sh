# RUN Synthetic Benchmark
algo_array=("allreduce" "decentralized" "relay")

for (( i=0; i<${#algo_array[@]}; i++ ));
do
    echo "Running with algo=${algo_array[$i]}"
    filename="./logs/exp2_${algo_array[$i]}.log"
    echo "Saving in $filename"
    python3 -m bagua.distributed.launch --nproc_per_node=8 synthetic_benchmark.py --algorithm ${algo_array[$i]} 2>&1 | tee $filename
done
