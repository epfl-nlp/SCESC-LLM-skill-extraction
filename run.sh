# !/bin/bash

# List of dataset names
dataset_names=("gnehm" "skillspan" "sayfullina" "green" "fijo" "kompetencer")

# List of prompt types
prompt_types=("ner" "extract")


# Outer loop for dataset names
for dataset_name in "${dataset_names[@]}"; do
    # Downloading dataset
    python main.py --knn --dataset_name "$dataset_name"

    # Inner loop for prompt types
    for prompt_type in "${prompt_types[@]}"; do
        # Running inference
        python main.py --run --shots 5 --knn --prompt_type "$prompt_type" --start_from_saved --dataset_name "$dataset_name" --model gpt-3.5-turbo
    done

    # Evaluate
    python main.py --eval --shots 5 --knn --prompt_type "$prompt_type" --dataset_name "$dataset_name" --model gpt-3.5-turbo
done
