#!/bin/bash

# 1 2 3 4 5 6
layers=(1 2 3 4 5 6 7 8)

# Run sequentially
for layer in "${layers[@]}"
do
    echo "Running with layer $layer..."
    python3 -u -m src.run  --num_layers="$layer"  --dataset='squirrel'   --k_plus=1 > "faber_squir_k1_layer${layer}.log" 2>&1
    echo "Finished seed $layer."
done

echo "All runs completed."
