# Rethinking Skill Extraction in the Job Market Domain using Large Language Models

## Introduction
This repo contains the code for the paper [***Rethinking Skill Extraction in the Job Market Domain using Large Language Models***](https://aclanthology.org/2024.nlp4hr-1.3/), published at the NLP4HR Workshop @ EACL2024.
Don't hesitate to contact us if you have questions!

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
```
@inproceedings{nguyen-etal-2024-rethinking,
    title = "Rethinking Skill Extraction in the Job Market Domain using Large Language Models",
    author = "Nguyen, Khanh  and
      Zhang, Mike  and
      Montariol, Syrielle  and
      Bosselut, Antoine",
    booktitle = "Proceedings of the First Workshop on Natural Language Processing for Human Resources (NLP4HR 2024)",
    month = mar,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.nlp4hr-1.3",
}
```
