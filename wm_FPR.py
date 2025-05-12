import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import copy
from contextlib import nullcontext

import sys
sys.path.append("/home/mingjia/med_watermark/synthid-text/src")
######## HF CACHE (LOAD BEFORE HF PACKAGES) ########
# os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

import torch
from transformers import LogitsProcessorList
from torch.cuda.amp import autocast
import statistics
import json
import yaml

# better bool flag type for argparse
from utils.data_loader import data_loader, human_text_loader
from utils.model_loader import model_loader_dist_shift, tokenizer_loader_SynthID, get_synthid_config
from synthid_text import detector_mean
from synthid_text import logits_processing
from synthid_text.synthid_generate import generate_responses, generate_g_values_and_mask

from watermark_processor_kgw import KGWWatermarkLogitsProcessor, KGWWatermarkDetector
from watermark_processor_medmark import MedWatermarkLogitsProcessor, MedWatermarkDetector

######## SEEDS ########

wm_keys = [15485863, 15485959, 15485867]
hash_key=42
random.seed(hash_key)
np.random.seed(hash_key)
torch.manual_seed(hash_key)
torch.cuda.manual_seed(hash_key)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def main(args):

    ######## LOAD MODELS ########

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scheme != "SynthID":
        model_short_name, _, tokenizer, tokenizer_trained, embed_matrix, _ = model_loader_dist_shift(args, device, detect_only=True)
    else:
        model_short_name, tokenizer = tokenizer_loader_SynthID(args)
    
    ######## LOAD DATASETS ########

    human_texts = human_text_loader(args, tokenizer)

    ######## TEXT GENERATION AND DETECTION ########

    ########## KGW evaluation ###########
    # kgw_gamma_list =  [0.1, 0.25, 0.5, 0.75, 0.9]
    # for gamma in kgw_gamma_list:
    #     args.gamma = gamma
    #     file_prefix = f"eval/FPR/KGW/len_{args.new_tokens}/{args.gamma}" 

    ########## MedMark evaluation ###########

    ckpt_list = [
        "0.1_4.0.pth",
        "0.1_4.5.pth",
        "0.1_5.0.pth",
        "0.1_5.5.pth",
        "0.1_6.0.pth",
        "0.1_6.5.pth"
    ]

    for path in ckpt_list:
        args.ckpt_path = "ckpt/" + path
        gamma_delta = path[:7]
        file_prefix = f"eval/FPR/MedMark/len_{args.new_tokens}/{gamma_delta}"

    ########## SynthID evaluation ###########

    # layer_list = [6]

    # for args.layer in layer_list:
    #     file_prefix = f"eval/FPR/SynthID/len_{args.new_tokens}/layer_{args.layer}"

        #### common code ####
        output_text_file = f"{file_prefix}/detail.json"
        output_eval_file = f"{file_prefix}/result.json"

        if args.log_generated_text:
            os.makedirs(os.path.dirname(output_text_file), exist_ok=True)
        
        if args.scheme == "KGW": 
            watermark_detector = KGWWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                device=device,
                                tokenizer=tokenizer,
                                gamma=args.gamma,
                                delta=args.delta,
                                normalizers=args.normalizers,
                                ignore_repeated_bigrams=args.ignore_repeated_bigrams)
        elif args.scheme == "MedMark":
            medmark_params = dict(
                vocab=list(tokenizer.get_vocab().values()),
                device=device,
                ckpt_path=args.ckpt_path, 
                gamma=args.gamma,
                embed_matrix=embed_matrix * args.embed_scaler,
                tokenizer_test=None if model_short_name == 'mistral' else tokenizer,
                tokenizer_trained=tokenizer if model_short_name == 'mistral' else tokenizer_trained
            )
            watermark_detector = MedWatermarkDetector(**medmark_params,
                                normalizers=args.normalizers,
                                ignore_repeated_bigrams=args.ignore_repeated_bigrams)
        elif args.scheme == "SynthID":
            CONFIG = get_synthid_config(args, device) 
            synthid_processor = logits_processing.SynthIDLogitsProcessor(
                **CONFIG, top_k=args.top_k, temperature=args.sampling_temp
            )

        z_score_human_list = []
        # human_texts = json.load(open("/home/mingjia/med_watermark/synthid-text/notebooks/test.json"))

        for i in range(args.cnt_fpr):
            if args.scheme == 'SynthID':
                bias = 1e-10
                uwm_g_values, uwm_mask = generate_g_values_and_mask(human_texts[i], tokenizer, device, synthid_processor, CONFIG)
                uwm_weighted_mean_scores = detector_mean.weighted_mean_score(
                    uwm_g_values.cpu().numpy(), uwm_mask.cpu().numpy()
                )
                z_score_human_list.extend(uwm_weighted_mean_scores.tolist())
            else:
                bias = 0.001
                with torch.no_grad():
                    with autocast():
                        score_dict = watermark_detector.detect(human_texts[i])
                z_score = score_dict['z_score'].item()
                z_score_human_list.append(z_score)

        z_score_human_list.sort(reverse=True)
        
        fpr_0 = z_score_human_list[int(args.cnt_fpr * 0.001)] + bias
        fpr_1 = z_score_human_list[int(args.cnt_fpr * 0.01)] + bias

        ######## OUTPUTS ########
        eval_results = {
            'emp_thres_1%': fpr_1,
            'emp_thres_0.1%': fpr_0,
            'cnt_fpr': args.cnt_fpr,
            'new_tokens': args.new_tokens,
        }
        # print(z_score_human_list, eval_results)
        if args.log_generated_text:
            with open(output_eval_file, "w") as outfile:
                outfile.write(json.dumps(eval_results, indent=4))
            with open(output_text_file, "w") as outfile:
                outfile.write(json.dumps([z_score_human_list, human_texts], indent=4))

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")
    parser.add_argument("--config_file", type=str, default="MedMark.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    flat_config = {f"{group}.{key}": value for group, params in config.items() for key, value in params.items()}
    
    for key, value in flat_config.items():
        group, param = key.split('.')
        setattr(args, param, value)

    # for removing some columns to save space
    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    main(args)