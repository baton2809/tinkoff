#!/bin/bash

if [ -d "processed_dataset" ]; then 
    rm -rf processed_dataset
fi
if [ -d "trained_model" ]; then 
    rm -rf trained_model
fi
if [ -d "wandb" ]; then
    rm -rf wandb
fi

find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Очистка выполнена."

