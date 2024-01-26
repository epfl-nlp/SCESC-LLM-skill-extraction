import string
import re

def write_answer_extract(list_skills):
    # process list of extracted skills to write is as demonstration
    if len(list_skills) == 0:
        return "None"
    else:
        return "\n".join(list_skills)

def get_list_of_selections(model_output, sentence, prompt_type, args):
    if prompt_type == 'ner':
        return get_list_of_selections_ner(model_output, sentence, args)
    elif prompt_type == 'extract':
        return get_list_of_selections_extract(model_output, sentence, args)
    else:
        raise Exception('prompt type not supported')

def get_list_of_selections_extract(model_output, sentence, args):
    # model_output is a list of strings separated by \n. Sentence is the list of tokens of the original sentence.
    list_of_selections = ['O']*len(sentence)
    sentence = [token.lower() for token in sentence]
    if "None" in model_output or model_output == "":
        return list_of_selections
    model_output = [str(item) for item in model_output.split('\n')]
    for skill in model_output:
        skill_tokens = skill.lower().split()
        if skill_tokens[0] not in sentence:
            skill_tokens[0] = [token for token in sentence if skill_tokens[0] in token][0]
        skill_index = sentence.index(skill_tokens[0])
        list_of_selections[skill_index] = 'B'
        for i in range(1, len(skill_tokens)):
            list_of_selections[skill_index + i] = 'I'
    return list_of_selections

def get_list_of_selections_ner(model_output, sentence, args):
    sentence = ' '.join(sentence)
    if model_output != "":
        model_output, _, _ = postprocess_ner_prompt(sentence, model_output, args)
    list_of_selections = []
    model_output = model_output.split()
    in_span = False
    for token in model_output:
        if not in_span:
            if '@@' in token and '##' not in token:
                in_span = True
                list_of_selections.append("B")
                continue
            elif '@@' in token and '##' in token:
                list_of_selections.append("B")
                continue
            else:  
                list_of_selections.append("O")
                continue
        
        if in_span:
            if '##' in token:
                in_span = False
                list_of_selections.append("I")
                continue
            else:
                list_of_selections.append("I")
                continue

    return list_of_selections

def check_format_response(original, generated, prompt_type, args):
    """
    Check if the generated response is in the correct format. If not, return feedback to the user.
    """
    feedback = ''
    if prompt_type == 'ner':
        _, mismatched, extracted = postprocess_ner_prompt(original, generated, args)
        if mismatched: 
            feedback = "You didn\'t correctly replicate the given sentence. Make sure the sentence stays the same, even if there are no skills to highlight, including punctuation, spacing, and grammar mistakes. Don\'t add any extra words or punctuation to the sentence except for the ## and @@ tags. Don\'t add nor remove any space." 
            if len(extracted) > 0:
                extracted_str = ", ".join(extracted)
                feedback += f" Remember to kept the valid highlighted skills with tags '@@' and '##': {extracted_str}"

    elif prompt_type == 'extract':
        original = original.lower()
        if generated=="None":
            feedback = ''
        else:
            missing_skills = []
            correct_skills = []
            extracted_skills = generated.lower().split('\n')
            for skill in extracted_skills:
                if skill not in original:
                    missing_skills.append(skill)
                else:
                    correct_skills.append(skill)
            if len(missing_skills) > 0:
                if len(correct_skills) > 0:
                    extracted_correct_skills_str = ", ".join(correct_skills)
                    feedback = f"You have correctly extracted these skills: {extracted_correct_skills_str}. "
                extracted_missing_skills_str = ", ".join(missing_skills)
                feedback += f"The following skills you extracted are either absent or not written the same way as in the original sentence: {extracted_missing_skills_str}. Modify these skills to make sure to exactly replicate these skills from the input sentence with their original spellings and grammars, discard any of them if needed."
                if len(correct_skills) > 0:
                    extracted_correct_skills_str = ", ".join(correct_skills)
                    feedback += " Remember to keep the skills that you correctly extracted."
                feedback += " Provide them with one skill per line."
    return feedback

def extract_spans(sentence):
    pattern = r'@@(.*?)##'
    spans = re.findall(pattern, sentence)
    return spans

def postprocess_ner_prompt(original, generation, dataset_name):
    print("======= INSIDE POSTPROCESS =======")
    print(f"ORIGINAL: {original}")
    print(f"GENERATION: {generation}")

    puntuation_list = ['.', ',', '!', '?', ';', ':', '\'', '"', '/', '(', ')', '[', ']', '{', '}']

    # dataset specific rules
    if dataset_name != 'kompetencer' and generation.endswith("##") and generation[-3] in puntuation_list:
        if generation[-4] == ' ':
            generation = generation[:-4] + "##" + generation[-3]
        else:
            generation = generation[:-3] + "##" + generation[-3]
    elif dataset_name == 'kompetencer' and generation.endswith("##."):
        generation = generation[:-3] + " .##"
    if original[-1] not in puntuation_list and generation[-1] in puntuation_list:
        generation = generation[:-1]

    extracted_spans = extract_spans(generation)

    pattern = r"(\w|##)([.,!?;:')\]}\"\/](?:##)?)|((?:@@)?[.,!?;:'(\[{\"\/])(\w)"

    # add spaces around punctuation
    cleaned_generation = re.sub(pattern, r'\1 \2 \3 \4', generation)
    # remove duplicated spaces
    cleaned_generation = re.sub(r'\s+', ' ', cleaned_generation).rstrip()

    if len(original) > 1 and original[-1] in puntuation_list and original[-2] != ' ':
        generation = generation[:-1] 

    print(f"CLEANED: {cleaned_generation}")

    mismatched = False 

    original_fixed = []
    generation_fixed = []
    original_idx = 0
    generated_idx = 0

    while original_idx < len(original) and generated_idx < len(cleaned_generation):
        original_char = original[original_idx]
        generated_char = cleaned_generation[generated_idx]

        # Check if the characters match
        if original_char == generated_char:
            original_fixed.append(original_char)
            generation_fixed.append(generated_char)
            original_idx += 1
            generated_idx += 1

        else:
            # NER-style special characters
            if generated_char == "#" or generated_char == "@":
                generation_fixed.append(generated_char)
                generated_idx += 1
            
            # if there is a space in the generation
            elif generated_char == ' ':
                if original_char in puntuation_list and cleaned_generation[generated_idx + 1] == original_char \
                    or cleaned_generation[generated_idx - 1] in puntuation_list and cleaned_generation[generated_idx + 1] == original_char:
                    generation_fixed.append(cleaned_generation[generated_idx + 1])
                    original_fixed.append(original_char)
                    generated_idx += 2
                    original_idx += 1
               
                elif cleaned_generation[generated_idx - 1] in puntuation_list and \
                    cleaned_generation[generated_idx + 1] == '@' and cleaned_generation[generated_idx + 3] == original_char: 
                    generation_fixed.extend(['@', '@'])
                    generation_fixed.append(cleaned_generation[generated_idx + 3])
                    original_fixed.append(original_char)
                    generated_idx += 4
                    original_idx += 1 
 
                else:
                    mismatched = True
                    break
            
            elif generated_char in puntuation_list:
                if original_char == ' ' and original[original_idx + 1] == generated_char:
                    generation_fixed.append(' ')
                    generation_fixed.append(generated_char)
                    original_fixed.append(original_char)
                    original_fixed.append(original[original_idx + 1])
                    generated_idx += 1
                    original_idx += 2
                elif original_char in string.ascii_lowercase:
                    if cleaned_generation[generated_idx + 2] == original_char: # random punctuation assertion
                        generation_fixed.append(cleaned_generation[generated_idx + 2])
                        original_fixed.append(original_char)
                        generated_idx += 3
                        original_idx += 1
                    elif cleaned_generation[generated_idx + 2] == "@" and cleaned_generation[generated_idx + 4] == original_char:
                        generation_fixed.extend(['@', '@'])
                        generation_fixed.append(cleaned_generation[generated_idx + 4])
                        original_fixed.append(original_char) 
                        generated_idx += 5
                        original_idx += 1
                    else:
                        mismatched = True
                        break
                else:
                    mismatched = True
                    break

            # check for random spaces in original
            elif generated_char in string.ascii_lowercase and original_char == ' ': 
                if (cleaned_generation[generated_idx-4:generated_idx] == original[original_idx-4:original_idx]) \
                and (cleaned_generation[generated_idx:generated_idx+4] == original[original_idx+1:original_idx+5]):
                   generation_fixed.append(original_char)
                   original_idx += 1 
                else:
                    mismatched = True
                    break
                     
            # check for special characters, for e.g. a hyphen in-place of a space
            elif original_char not in string.ascii_lowercase and generated_char not in string.ascii_letters:
                generated_idx += 1
                original_idx += 1
            else:
                mismatched = True
                break
        

    print(original[original_idx:])
    original_fixed.extend(original[original_idx:])
    generation_fixed.extend(cleaned_generation[generated_idx:])
   
    generated_fixed_str = ''.join(generation_fixed)

    if len(original.split()) != len(generated_fixed_str.split()):
        mismatched = True

    extracted_spans = [ent for ent in extracted_spans if ent in original]

    print(f"UPDATED: {generated_fixed_str}")
    print(f"mismatched: {mismatched}")
    print("================================")
    return generated_fixed_str, mismatched, extracted_spans