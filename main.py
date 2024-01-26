import argparse
import os
import random
import json

import pandas as pd

import torch
from preprocess import load_skills_data, preprocess_dataset
from run import run_openai
from evaluate_src import eval
from demo_retrieval import embed_demo_dataset

random.seed(1234)

def download(args, split):
    dataset = load_skills_data(args.dataset_name, split)
    dataset.to_json(args.raw_data_dir + '/' + split + '.json', orient='records', indent=4, force_ascii=False)
    print(f'Saved {args.dataset_name} dataset to {args.raw_data_dir}, with {len(dataset)} examples.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='gnehm', help='Dataset name to use. Default is gnehm. Options are green, skillspan, fijo, sayfullina, kompetencer')
    parser.add_argument('--prompt_type', type=str, default='ner')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--raw_data_dir', type=str, default='data/annotated/raw/')
    # run parameters
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--knn', action='store_true', help='Use KNN retrieval instead of random sampling for demonstrations')
    parser.add_argument('--process', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_from_saved', action='store_true', help='Start from saved results instead of running inference again.')
    parser.add_argument('--exclude_empty', action='store_true', help='Exclude examples that have no skills in them.')
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--positive_only', action='store_true', help='whether to include only positive samples from the dataset')
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--sample', type=int, default=0, help='number of samples to perform inference on, for debugging.')
    parser.add_argument('--exclude_failed', action='store_true', help='whether to exclude previous failed attempt') 
    
    args = parser.parse_args()
    args.save_path = args.save_path + '/' + args.model + '/' + args.dataset_name + '_' + args.prompt_type + '_' + str(args.shots) + '-shots.json'

    if args.knn:
        args.save_path = args.save_path.replace('.json', '_knn.json')
    args.raw_data_dir = args.raw_data_dir + args.dataset_name + '/'
    args.processed_data_dir = args.raw_data_dir.replace('raw', 'processed')
    args.embeddings_dir = args.raw_data_dir.replace('raw', 'embeddings')
    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
    # if args.model not in ["gpt-3.5-turbo", "gpt-4"] or "Llama-2-7b" not in args.model:
    #     raise Exception("model not supported")
    
    if args.prompt_type == 'ner': # Gold is 'sentence_with_tags'
        args.gold_column = 'sentence_with_tags'
    elif args.prompt_type == 'extract': # Gold is 'list_extracted_skills'
        args.gold_column = 'list_extracted_skills'
    return args


def main():
    args = parse_args()
    
    # Download dataset if not already stored
    if not os.path.exists(args.raw_data_dir):
        os.makedirs(args.raw_data_dir)
    for split in ['train', 'test']:
        if not os.path.exists(args.raw_data_dir + '/' + split + '.json'):
            print(f'Downloading {args.dataset_name} dataset, {split} split...')
            download(args, split)

    # Process the dataset
    for split in ['train', 'test']:
        processed_path = args.processed_data_dir + split + '.json'
        if not os.path.exists(processed_path) or args.process:
            print(f'Processing {args.dataset_name} dataset...')
            dataset = preprocess_dataset(args, split)
    args.data_path = args.processed_data_dir + '/test.json'
    args.demo_path = args.processed_data_dir + '/train.json'

    if args.knn:
        # Embed the dataset (train for demos, and test)
        for split in ['train', 'test']:
            emb_save_path = args.embeddings_dir + '/' + split + '.pt'
            if not os.path.exists(emb_save_path):
                print(f'Generating {split} set embeddings for {args.dataset_name} dataset...')
                source_dataset = json.load(open(args.processed_data_dir + split + '.json'))
                if len(source_dataset) > 500 and split == 'train':
                    source_dataset = random.sample(source_dataset, 500)
                dataset_texts = [sample["sentence"] for sample in source_dataset]
                dataset_ids = [sample["id"] for sample in source_dataset]
                dataset_embed = embed_demo_dataset(dataset_texts, args.dataset_name)
                embeddings_dict = {'embeddings': dataset_embed, 'ids': dataset_ids}
                torch.save(embeddings_dict, emb_save_path)
                print(f'Saved {args.dataset_name} dataset embeddings to {emb_save_path}.')

    if args.run:
        # Load dataset
        dataset = pd.read_json(args.data_path)
        if args.exclude_empty:
            dataset['has_item'] = dataset.apply(lambda row: len(row['skill_spans'])>0, axis=1)
            dataset = dataset[dataset['has_item'] == True]
            dataset.drop(columns=['has_item'], inplace=True)
        print(f"Loaded {len(dataset)} examples from {args.dataset_name} dataset.")
        if len(dataset['id']) != len(set(dataset['id'].values.tolist())):
            raise Exception("The ids are not unique")
        # Run inference
        if args.sample != 0:
            sample_size = min(args.sample, len(dataset))
            dataset = dataset.sample(sample_size, random_state=1450).reset_index(drop=True)
        run_openai(dataset, args)
    
    if args.eval:
        print(f'Evaluating {args.save_path}...')
        all_metrics = eval(args.save_path)
        print(all_metrics)        

if __name__ == "__main__":
    main()