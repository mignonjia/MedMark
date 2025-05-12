import json
from datasets import load_dataset

def data_loader(args, model_short_name):
    # construct the collator
    
    if args.dataset_name == "HealthSearchQA":

        entrys = json.load(open(f"data/HealthSearchQA/QA_{args.dataset_split}.json"))

        with open(f"data/{model_short_name}_template/qa_template.txt") as f:
            lines = f.readlines()
            prompt_template = "".join(lines)

        data = [prompt_template.replace('TEMPLATE', val.strip()) for val in entrys]

    if args.dataset_name == "radiology":

        entrys = json.load(open(f"data/radiology/mid_target.json", "r"))
        
        data = []
        
        with open(f"data/{model_short_name}_template/radiology_summ_template.txt") as f:
            lines = f.readlines()
            prompt_template = "".join(lines)
        
        # data = [prompt_template.replace('FINDING_TEMPLATE', val) for val in data]
        for entry in entrys:
            query = prompt_template.replace('FINDING_TEMPLATE', entry['finding']).replace('INDICATION_TEMPLATE', entry['indication'])
            data.append(query)

    if args.dataset_name == "clinical_notes":

        entrys = json.load(open(f"data/clinical_notes/entrys.json", "r"))
        data = []
        
        with open(f"data/{model_short_name}_template/clinical_notes_template.txt") as f:
            lines = f.readlines()
            prompt_template = "".join(lines)
        
        for entry in entrys:
            query = prompt_template.replace('DISCHARGE_PLACE', entry['note']).replace('QUESTION_PLACE', entry['question'])
            data.append(query)


    return data

def human_text_loader(args, tokenizer):
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    long_answers = dataset['train']['long_answer']

    text = ''.join(long_answers[:34000])
    input_ids = tokenizer(text, return_tensors='pt')["input_ids"]

    chunk_size = args.new_tokens

    input_ids = input_ids[:, :(chunk_size * args.cnt_fpr)]
    human_texts = []
    for i in range(args.cnt_fpr):
        chunk_ids = input_ids[:, i * chunk_size: (i + 1) * chunk_size]
        human_text = tokenizer.decode(chunk_ids.squeeze(), skip_special_tokens=True)
        human_texts.append(human_text)

    return human_texts

def get_words_list():
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    long_answers = dataset['train']['long_answer']

    text = ' '.join(long_answers[:1000])
    words = list(set(text.split(" ")))
    
    return words