from datasets import load_dataset
import pandas as pd

# this method just loads the dataset and drops the pos column
# can be used to get other splits of the dataset as well
def load_skills_data(dataset_name, split):
    dataset_name = "jjzha/" + dataset_name
    dataset = load_dataset(dataset_name)
    dataset = pd.DataFrame(dataset[split])
    try:
        # for gnehm dataset
        dataset.drop(columns=['pos', 'idx'], inplace=True)
    except:
        pass
    dataset['idx'] = dataset.index
    dataset.rename(columns={'idx': 'id'}, inplace=True)
    return dataset

def uniformize_skills_column_per_row(row, dataset_name):
    tags_skill = row['tags_skill']
    if dataset_name in ['gnehm', 'fijo', 'sayfullina']:
        fixed_tags_skills = list(map(lambda item: item[0] if "-" in item else 'O', tags_skill))
    if dataset_name in ['skillspan', 'kompetencer']:
        fixed_tags_skills = tags_skill
    if dataset_name == 'green':
        fixed_tags_skills = list(map(lambda item: item.replace('-SKILL', '') if "SKILL" in item else 'O', tags_skill))
    return fixed_tags_skills

# applies the uniformize_skills_column_per_row method to the whole dataset
def uniformize_skills_column(dataset, dataset_name):
    dataset['tags_skill_clean'] = dataset.apply(lambda x: uniformize_skills_column_per_row(x, dataset_name), axis=1)
    return dataset

def bio_tags_to_spans(tag_sequence):
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    spans = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            print(f"Unrecognized BIO tag: {bio_tag}")
            bio_tag = "O" 
            #raise ValueError(f"Unrecognized BIO tag: {bio_tag}")
        conll_tag = string_tag[2:]
        if bio_tag == "O":
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

# adds beginning and end tags to the words that are tagged as skills
def add_tags_to_words(words, spans, begin_tag='@@', end_tag='##'):
    tokens = words.copy()
    for span in spans:
        start, end = span[1][0], span[1][1]

        tokens[start] = begin_tag + tokens[start]
        tokens[end] = tokens[end] + end_tag
    return tokens

def extract_skill_tokens(words, spans):
    skills_list = []
    for span in spans:
        start, end = span[1][0], span[1][1]
        skills_list.append(' '.join(words[start:end+1]))
    return skills_list

def add_golden_answer_column(dataset):
    dataset['tokens_with_tags'] = dataset.apply(lambda row: add_tags_to_words(row['tokens'], row['skill_spans']), axis=1)
    dataset = concat_tokens(dataset)
    dataset['list_extracted_skills'] = dataset.apply(lambda row: extract_skill_tokens(row['tokens'], row['skill_spans']), axis=1)
    return dataset


# just a concatenation method on the list of tokens
def concat_tokens(dataset):
    dataset['sentence'] = dataset.apply(lambda row: ' '.join(row['tokens']), axis=1)
    dataset['sentence_with_tags'] = dataset.apply(lambda row: ' '.join(row['tokens_with_tags']), axis=1)
    return dataset


def drop_long_examples(dataset, max_length=200):
    dataset = dataset[dataset['tokens'].map(len) < max_length]
    return dataset

# this method performs all the required preprocessing on the dataset
def preprocess_dataset(args, split):
    dataset = pd.read_json(args.raw_data_dir + split + '.json') # TODO check size of dataset
    
    # uniformize format of B I O annotations
    dataset = uniformize_skills_column(dataset, args.dataset_name)
    dataset = drop_long_examples(dataset)

    # get spans from B I O annotations
    dataset['skill_spans'] = dataset.apply(lambda row: bio_tags_to_spans(row['tags_skill_clean']), axis=1)
    
    #dataset = dataset.sample(frac = 1, random_state=1450).reset_index(drop=True)
    dataset = add_golden_answer_column(dataset)  

    # save processed dataset
    save_path = args.processed_data_dir + split + '.json'
    dataset.to_json(save_path, orient='records', indent=4, force_ascii=False)
    print(f'Saved {args.dataset_name} dataset to {save_path}, with {len(dataset)} examples.')
    return dataset
