PROMPT_TEMPLATES = {
    "all": {
        "system": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from a job description. Replicate the sentence and highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job description. Extract all the skills and competencies that are required from the candidate as a list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "gnehm": {
        "system": "You are an expert human resource manager in information and communication technology (ICT) from Germany. You need to analyse skills required in German job offers.",
        "instruction": {
            "ner": "You are given an extract from a job advertisement in German. Highlight all the IT/Technology skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job advertisement in German. Extract all the IT/Technology skills and competencies that are required from the candidate as list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "skillspan": {
        "system": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from a job posting. Highlight all the skills, knowledges, and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job posting. Extract all the skills, knowledges, and competencies that are required from the candidate as list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "sayfullina": {
        "system": "You are an expert human resource manager. You need to detect and analyse soft skills required in job offeres",
        "instruction": {
            "ner": "You are given a sentence from a job advertisement. Highlight all the soft skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job advertisement. Extract all the soft skills and competencies that are required from the candidate as list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "fijo": {
        "system": "You are an expert human resource manager in the insurance industry in France. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from an insurance job description in French. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from an insurance job description in French. Extract all the skills and competencies that are required from the candidate as list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "green": {
        "system": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from a job description in various fields like IT, finance, healthcare, and sales. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.  If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job descriptionin various fields like IT, finance, healthcare, and sales. Extract all the skills and competencies that are required from the candidate as list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "kompetencer": {
        "system": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from a job description in Danish. Highlight all the skills, knowledges, and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job description in Danish. Extract all the skills, knowledges, and competencies that are required from the candidate as list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
}
