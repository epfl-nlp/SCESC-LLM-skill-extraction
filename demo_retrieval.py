import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

models = {'green': 'jjzha/jobbert-base-cased',# English
        'skillspan': 'jjzha/jobbert-base-cased',# English
        'fijo': 'camembert-base', # French
        'sayfullina': 'jjzha/jobbert-base-cased', # English
        'kompetencer': 'jjzha/dajobbert-base-uncased', # danish
        'gnehm': 'agne/jobBERT-de'} # german

def embed_demo_dataset(demo_dataset_texts, dataset):
    """
    Embed the demo dataset
    demo_data_file: path to the demo dataset json file
    """
    model = AutoModel.from_pretrained(models[dataset])
    tokenizer = AutoTokenizer.from_pretrained(models[dataset])
    # Tokenize and batch the sentences
    tokenized_demo_dataset = tokenizer(demo_dataset_texts, padding=True, truncation=True, return_tensors="pt", max_length=128, return_attention_mask=True)
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(demo_dataset_texts), batch_size)):
        batch_input = {k: v[i:i+batch_size] for k, v in tokenized_demo_dataset.items()}
        with torch.no_grad():
            output = model(**batch_input, output_hidden_states=True, return_dict=True)
            batch_embeddings = output.pooler_output
            embeddings.append(batch_embeddings)
    # Concatenate the embeddings to get the final result
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def knn_demo_retrieval(test_sentence_id, args):
    """
    KNN retrieval from the demonstration dataset for ICL
    input: the input text (inference sample)
    nb_shots: number of demonstration samples to retrieve
    demo_data_file: path to the demo dataset json file
    """
    # TODO do this only once
    demo_embed = torch.load(args.embeddings_dir + '/train.pt')
    test_embed = torch.load(args.embeddings_dir + '/test.pt')

    demo_embeddings = demo_embed['embeddings']
    demo_ids = demo_embed['ids']
    input_embed = [emb for emb, id in zip(test_embed['embeddings'], test_embed['ids']) if id == test_sentence_id][0]
    cosine_scores = [(similarity, index) for similarity, index in zip(cosine_similarity(demo_embeddings, input_embed.unsqueeze(0)), demo_ids)]
    sorted_cosine_scores = sorted(cosine_scores, key=lambda x: x[0], reverse=True)
    sorted_indices = [score[1] for score in sorted_cosine_scores]
    return sorted_indices
