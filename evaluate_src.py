import string
import re

import pandas as pd
import evaluate
seqeval = evaluate.load("seqeval")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))    

def calc_avg_score(dataset, filtered = False):    
    skillscore_values = dataset['skill_level_score'].apply(lambda x: list(x.values())).apply(pd.Series)

    skillscore_values.columns = ['skillevel_precision', 'skillevel_recall', 'skillevel_f1']
    if filtered:
        skillscore_values.columns = ['filtered_skillevel_precision', 'filtered_skillevel_recall', 'filtered_skillevel_f1']

    # Compute the mean of each element across all rows
    #mean_seqeval_values = seqeval_values.mean().apply(lambda x: round(x, 3)).to_dict()
    mean_skillscore_values = skillscore_values.mean().apply(lambda x: round(x, 3)).to_dict()
    return mean_skillscore_values

def extract_entities_from_bio(tokens, bio_tags):
    entities = []
    current_entity = []

    for token, bio_tag in zip(tokens, bio_tags):
        bio_type = bio_tag.split('-')[-1] if '-' in bio_tag else bio_tag

        if bio_type == 'B':
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [token]

        elif bio_type == 'I':
            current_entity.append(token)

        elif bio_type == 'O':
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return entities

def skill_level_metrics(df):
    # for each span of row['skill_spans'], check if at least one word was retrieved.
    span_lists = df['skill_spans'].values.tolist()
    predictions = df['list_of_selection'].values.tolist() 
    references = df['tags_skill_clean'].values.tolist()

    nb_correct = 0
    nb_gold_skills = 0
    nb_pred_skills = 0

    for span_list, pred, ref in zip(span_lists, predictions, references):
        nb_gold_spans = len(span_list)
        nb_pred_spans = pred.count('B')
        if nb_gold_spans != 0:
            predicted_skills_indices = [index for index, pred_tag in enumerate(pred) if pred_tag!='O']
            for span in span_list:
                span = range(span[1][0], span[1][1]+1)
                # test if element of predicted_skills_indices is in span
                if any(found_item in span for found_item in predicted_skills_indices):
                    nb_correct += 1
            nb_gold_skills += nb_gold_spans
        nb_pred_skills += nb_pred_spans  

    # Calculate Precision and Recall
    precision = nb_correct / (nb_pred_skills)
    recall = nb_correct / (nb_gold_skills)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {'skillevel_precision': precision, 'skillevel_recall': recall, 'skillevel_f1': f1_score}


def entities_level_metrics(df, prompt_type):
    prediction_texts = df['model_output'].values.tolist()
    # bio_predictions = df['list_of_selection'].values.tolist() 
    sequences = df['tokens'].values.tolist()
    references = df['list_extracted_skills'].values.tolist()

    nb_correct = 0
    nb_gold_skills = 0
    nb_pred_skills = 0

    for seq, p_text, ref in zip(sequences, prediction_texts, references):
        p_text = p_text.split("\n\n")[0]
        if prompt_type == 'ner':
            def extract_entities_between_markers(sentence, start_marker='@@', end_marker='##'):
                pattern = fr'{re.escape(start_marker)}(.*?){re.escape(end_marker)}'
                entities = re.findall(pattern, sentence)
                return entities if entities else []
            
            pred = extract_entities_between_markers(p_text)

        elif prompt_type == 'extract':
            pred = p_text.split("\n")
            if p_text == "None":
                pred = []
            else:
                pred = list(set([normalize_answer(en) for en in p_text.split("\n")]))
        for en in pred:
            if en in ref:
                nb_correct += 1
            nb_pred_skills += 1
        nb_gold_skills += len(ref) 

    # Calculate Precision and Recall
    precision = nb_correct / (nb_pred_skills + 1e-10)
    recall = nb_correct / (nb_gold_skills + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {'entitylevel_precision': precision, 'entitylevel_recall': recall, 'entitylevel_f1': f1_score}

def eval(save_path):    
    df = pd.read_json(save_path)
    all_metrics = {}
    print("Removing examples where the number of spans and the number of predictions don't match:", len(df[df['list_of_selection'].apply(len) != df['tags_skill_clean'].apply(len)]))
    df = df[df['list_of_selection'].apply(len) == df['tags_skill_clean'].apply(len)]
    
    seqeval_score = seqeval.compute(predictions=df['list_of_selection'].values.tolist(), references=df['tags_skill_clean'].values.tolist()) 
    seqeval_score = {k.replace('overall_', 'seqeval_'):v for k, v in seqeval_score.items() if k in ['overall_precision', 'overall_recall', 'overall_f1']}
    all_metrics.update(seqeval_score)

    skilllevel_score = skill_level_metrics(df)
    all_metrics.update(skilllevel_score)

    return all_metrics