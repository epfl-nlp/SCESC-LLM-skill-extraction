# Rethinking Skill Extraction in the Job Market Domain using Large Language Models

## Introduction
This repo contains the code for the paper ***Rethinking Skill Extraction in the Job Market Domain using Large Language Models***, to be appeared in NLP4HR Workshop @ EACL2024.

## Usage
End-to-end experiments can be run with the following command
```bash
sh run.sh
```

## Datasets
Datasets used for experiments can be found [here](https://huggingface.co/jjzha). Additionally, you can download the processed annotation model by running the following command
```python
python main.py --knn --dataset_name $DATASET_NAME
```

## Running and Evaluation
Create an `api_key.py` and put your OpenAI API key under the variable `API_KEY`. Afterwards, you can run the experiments and evaluate the results using the following commands
```python
python main.py --run --shots $NUM_SHOTS --knn --prompt_type $PROMPT_TYPE [--start_from_saved] [--exclude_empty] [--positive_only] --dataset_name $DATASET_NAME --model $MODEL

python main.py --eval --shots $NUM_SHOTS --knn --prompt_type $PROMPT_TYPE --dataset_name $DATASET_NAME --model $MODEL
```
## Citation
