#!/bin/bash

# Define seeds to run 100 200 300 400 500 600
seeds=( 700 800 900 1000)

# Run sequentially
for seed in "${seeds[@]}"
do
    echo "Running with seed $seed..."
    python3 -u -m src.run --seed="$seed" > "keep_chame${seed}.log" 2>&1
    echo "Finished seed $seed."
done

echo "All runs completed."
