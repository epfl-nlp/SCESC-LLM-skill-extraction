import openai
import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
import json
import os
import random
from demo_retrieval import knn_demo_retrieval
from postprocess import check_format_response, postprocess_ner_prompt, get_list_of_selections

random.seed(1234)
np.random.seed(1234)

# this section sets the necessary api key to use the openai api
import sys
from api_key import API_KEY
from prompt_template import PROMPT_TEMPLATES
openai.api_key = API_KEY

def write_answer_extract(list_skills):
    # process list of extracted skills to write is as demonstration
    if len(list_skills) == 0:
        return "None"
    else:
        return "\n".join(list_skills)

def get_knn_demonstrations(demos_files, test_sentence_id, args):
    shots = min(len(demos_files), args.shots)
    demos_ids = knn_demo_retrieval(test_sentence_id, args)
    demos = []
    for i in range(len(demos_ids)):
        if len(demos) == shots:
            break
        demos.extend(sample for sample in demos_files if sample['id'] == demos_ids[i])
    return demos

def get_prompt(dataset, args, id, all_demos):
    instruction_field = args.dataset_name
    instruction = PROMPT_TEMPLATES[instruction_field]['instruction'][args.prompt_type]
    # TODO have specify prompt template for all datasets
    messages = [{"role": "system", "content": instruction}]
    
    row_index = dataset[dataset['id'] == id].index[0]
    row = dataset.iloc[row_index]

    indexes = dataset['id'].values.tolist()
    indexes.remove(id)

    if args.knn:
        positive_demos_knn = get_knn_demonstrations(all_demos[0], row['id'], args)
        negative_demos_knn = get_knn_demonstrations(all_demos[1], row['id'], args)
        demos = positive_demos_knn + negative_demos_knn
        random.shuffle(demos)
    else:
        demos = all_demos[2]

    for example in demos:
        question = "Sentence: " + str(example['sentence'])
        messages.append({"role": "user", "content": question})
        if args.prompt_type == 'extract':
            answer = write_answer_extract(example['list_extracted_skills'])
        else:
            answer = str(example['sentence_with_tags'])
        messages.append({"role": "assistant", "content": answer})

    question = "Sentence: " + row['sentence']
    messages.append({"role": "user", "content": question})

    return messages

def run_openai(dataset, args):
    if os.path.exists(args.save_path) and args.start_from_saved:
        df = pd.read_json(args.save_path)
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        df = pd.DataFrame(columns= list(dataset.columns) + ['model', 'prompt', 'model_output', 'list_of_selection'])
    print(f'saving to {args.save_path}')
        
    ids_all = dataset['id']
    ids_done = df['id']
    ids_left = list(set(ids_all) - set(ids_done))

    failed_sentences = set()
    if args.exclude_failed:
        failed_path = f"failed_extraction_{args.prompt_type}_{args.shots}_shot"
        if args.knn:
            failed_path += "_knn"
        failed_path += ".json"
        with open(failed_path, "r") as readfile:
            for line in readfile:
                instance = json.loads(line)
                failed_sentences.add(instance['sentence'])

    # sample demos from train set
    demos_dataset = json.load(open(args.processed_data_dir + 'train.json'))
    demos_with_skills = [sample for sample in demos_dataset if len(sample['skill_spans']) > 0]
    demos_without_skills = [sample for sample in demos_dataset if len(sample['skill_spans']) == 0]

    negative_shots = min(len(demos_without_skills), args.shots)
    if args.positive_only:
        fixed_demos = random.sample(demos_with_skills, args.shots * 2)
    else:
        fixed_demos = random.sample(demos_with_skills, args.shots) + random.sample(demos_without_skills, negative_shots)
    random.shuffle(fixed_demos)
    all_demos = (demos_with_skills, demos_without_skills, fixed_demos)
    
    for id in tqdm(ids_left,total=len(ids_left)):
        index_sample = dataset[dataset['id'] == id].index[0]
        row = dataset.iloc[index_sample]
        if row['sentence'] in failed_sentences:
            continue
        row_to_save = {}
        for key, value in row.items():
            row_to_save[key] = value
        messages = get_prompt(dataset, args, id, all_demos)
        response = openai.ChatCompletion.create(model=args.model, messages=messages, temperature=0)
        model_output = response['choices'][0]['message']['content']

        feedback = check_format_response(row['sentence'], model_output, args.prompt_type, args)
        trys_count = 0
        if feedback != '':
            print("######################### INCORRECT FORMAT DETECTED ##############################")
            print(feedback)
            print("---- Original Sentence")
            print(row['sentence'])
            print("---- Extraction")
            print(model_output)
            # If the model fails to generate the output correctly, try again up to 5 times
            # update the prompt with a new message targeting the specific issue
            while feedback != '' and trys_count < 3:
                print("Re-trying...", str(trys_count))
                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "user", "content": feedback})
                response = openai.ChatCompletion.create(model=args.model, messages=messages, temperature=0)
                model_output = response['choices'][0]['message']['content']
                feedback = check_format_response(row['sentence'], model_output, args.prompt_type, args)
                print("#### New feedback")
                print(feedback)
                print("---- Revised Extraction")
                print(model_output)
                trys_count += 1
        if trys_count == 3:
            failed_path = f"failed_extraction_{args.prompt_type}_{args.shots}_shot"
            if args.knn:
                failed_path += "_knn"
            failed_path += ".json"
            with open(failed_path, "a") as outfile:
                json.dump({"dataset": args.dataset_name, "sentence": row['sentence'], "extracted_skills": model_output.split('\n')}, outfile)
                outfile.write('\n')
            model_output = ""
        list_of_selections = get_list_of_selections(model_output, row['tokens'], args.prompt_type, args)
        if args.prompt_type == 'ner' and trys_count < 3:
            model_output, _, _ = postprocess_ner_prompt(row['sentence'], model_output, args)
    
        row_to_save['model'] = args.model
        row_to_save['prompt'] = messages
        row_to_save['model_output'] = model_output
        
        row_to_save['list_of_selection'] = list_of_selections
        df.loc[len(df)] = row_to_save
        df.to_json(args.save_path, orient='records', indent=4, force_ascii=False)