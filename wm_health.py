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
import time
# better bool flag type for argparse
from utils.data_loader import data_loader
from utils.model_loader import model_loader_dist_shift, model_loader_SynthID, model_loader_STS, get_gen_params, last_token_pool
from synthid_text import detector_mean
from synthid_text import logits_processing
from synthid_text.synthid_generate import generate_responses, generate_g_values_and_mask

from watermark_processor_kgw import KGWWatermarkLogitsProcessor, KGWWatermarkDetector
from watermark_processor_medmark import MedWatermarkLogitsProcessor, MedWatermarkDetector

######## SEEDS ########

wm_keys = [15485863]
gen_seed=42
random.seed(gen_seed)
np.random.seed(gen_seed)
torch.manual_seed(gen_seed)
torch.cuda.manual_seed(gen_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
def main(args):

    ######## LOAD MODELS ########

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_STS = device

    if args.scheme == "SynthID":
        model_short_name, model_wm, model_uwm, tokenizer_test, CONFIG, terminators = model_loader_SynthID(args, device)
    else:
        model_short_name, model, tokenizer_test, tokenizer_trained, embed_matrix, terminators = model_loader_dist_shift(args, device)
    
    # The `tokenizer_trained` is always the Mistral tokenizer. 
    # We need it because we trained gamma/delta based on Mistral tokenizer. 
    # If we test on other models, we need to Mistral tokenizer to compute gamma/delta.

    model_STS, tokenizer_STS = model_loader_STS(args, device_STS)

    ######## LOAD DATASET ########

    data = data_loader(args, model_short_name)

    total_time = 0
    start_time = time.time()

    ######## TEXT GENERATION AND DETECTION ########

    ########## KGW evaluation ###########
    # gamma = '0.1'
    # args.gamma = 0.1
    # if args.new_tokens == 100:
    #     kgw_delta_list = ['3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0']
    # elif args.new_tokens == 50:
    #     kgw_delta_list = ['3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0']
    # elif args.new_tokens == 200:
    #     kgw_delta_list = ['3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5'] 

    # for delta in kgw_delta_list:
    #     args.delta = float(delta)
    #     file_prefix = f"eval/{args.dataset_name}/{args.scheme}/len_{args.new_tokens}_temp_{args.sampling_temp}" 
    #     file_post = f"{args.gamma}_{args.delta}_{model_short_name}"

    ########### MedMark Evaluation ###########
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
        file_prefix = f"eval/{args.dataset_name}/{args.scheme}/len_{args.new_tokens}_temp_{args.sampling_temp}"
        file_post = f"{gamma_delta}_{model_short_name}"

    ########### SynthID ###########
    # params_list = [[50, 0.5], [50, 1.0], [100, 0.5], [100, 1.0], [200, 0.5], [200, 1.0]]

    # for args.new_tokens, args.sampling_temp in params_list:
    #     args.layer = len(CONFIG["keys"])
    #     args.H = CONFIG["ngram_len"] - 1
    #     args.num_leaves = CONFIG["num_leaves"]
        
    #     file_prefix = f"eval/{args.dataset_name}/{args.scheme}/len_{args.new_tokens}_temp_{args.sampling_temp}"
    #     file_post = f"num_leaves_{args.num_leaves}_layer_{args.layer}_{model_short_name}"

        #### common code start ####

        output_text_file = f"{file_prefix}/text/{file_post}.json_pp"
        output_eval_file = f"{file_prefix}/{file_post}.json"

        # Remove existing output_text_file if it exists
        if os.path.exists(output_text_file):
            os.remove(output_text_file)
            
        z_score_list, STS_list = [], []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) # function for STS

        if args.log_generated_text:
            os.makedirs(os.path.dirname(output_text_file), exist_ok=True)
        
        ########### GET WATERMARK PROCESSORS ###########
        wm_key = wm_keys[args.wm_key]

        if args.scheme == "KGW": 
            kwg_params = dict(
                vocab=list(tokenizer_test.get_vocab().values()),
                device=device,
                gamma=args.gamma,
                delta=args.delta,
                hash_key=wm_key,
                context_window=args.context_window
            )
            watermark_processor = KGWWatermarkLogitsProcessor(**kwg_params)
            watermark_detector = KGWWatermarkDetector(**kwg_params,
                                tokenizer=tokenizer_test, 
                                normalizers=args.normalizers,
                                ignore_repeated_bigrams=args.ignore_repeated_bigrams)
        elif args.scheme == "MedMark":
            medmark_params = dict(
                vocab=list(tokenizer_test.get_vocab().values()),
                device=device,
                ckpt_path=args.ckpt_path, 
                gamma=args.gamma,
                embed_matrix=embed_matrix * args.embed_scaler,
                tokenizer_test=None if model_short_name == 'mistral' else tokenizer_test, # if mistral, then tokenizer_test = tokenizer_trained, and pass None here
                tokenizer_trained=tokenizer_trained, 
                quantize=False,
                hash_key=wm_key,
                context_window=args.context_window
            )
            watermark_processor = MedWatermarkLogitsProcessor(**medmark_params) 
            watermark_detector = MedWatermarkDetector(**medmark_params,
                                normalizers=args.normalizers,
                                ignore_repeated_bigrams=args.ignore_repeated_bigrams)
        elif args.scheme == "SynthID":
            synthid_processor = logits_processing.SynthIDLogitsProcessor(
                **CONFIG, top_k=args.top_k, temperature=args.sampling_temp
            )

        for i in tqdm(range(0, len(data), args.batch_size)):

            cur_data = data[i:(i+args.batch_size)]
            inputs = tokenizer_test(cur_data, return_tensors='pt', padding=True).to(device)
            prefix_len = inputs['input_ids'].shape[1]
            
            gen_params = get_gen_params(args, terminators)
            
            ######## GENERATION ########

            with torch.no_grad():
                with autocast():
                    torch.manual_seed(gen_seed)
                    if args.scheme == 'SynthID':
                        uwm_g_values, uwm_mask, output_no_wm = generate_responses(model_uwm, tokenizer_test, inputs, gen_params, synthid_processor, CONFIG)
                    else: # MedMark, KGW
                        output_no_wm = model.generate(**inputs, **gen_params)
                    
                    torch.manual_seed(gen_seed) 
                    if args.scheme == 'SynthID':
                        wm_g_values, wm_mask, output_w_wm = generate_responses(model_wm, tokenizer_test, inputs, gen_params, synthid_processor, CONFIG)
                        # run detection here
                        wm_weighted_mean_scores = detector_mean.weighted_mean_score(
                            wm_g_values.cpu().numpy(), wm_mask.cpu().numpy()
                        )
                        z_score_cur_batch = wm_weighted_mean_scores.tolist()
                        z_score_list.extend(z_score_cur_batch)
                    else: # MedMark, KGW
                        output_w_wm = model.generate(**inputs, **gen_params,
                                        logits_processor=LogitsProcessorList([watermark_processor]))

                    decoded_output_no_wm = tokenizer_test.batch_decode(output_no_wm[:,prefix_len:], skip_special_tokens=True) #  
                    decoded_output_w_wm = tokenizer_test.batch_decode(output_w_wm[:,prefix_len:], skip_special_tokens=True)   

                    batch_dict_w_wm = tokenizer_STS(decoded_output_w_wm, padding=True, truncation=True, return_tensors="pt").to(device_STS)
                    embed_wm = model_STS(**batch_dict_w_wm)
                    embed_wm = last_token_pool(embed_wm.last_hidden_state, batch_dict_w_wm["attention_mask"])
                    batch_dict_no_wm = tokenizer_STS(decoded_output_no_wm, padding=True, truncation=True, return_tensors="pt").to(device_STS)
                    embed_no_wm = model_STS(**batch_dict_no_wm)
                    embed_no_wm = last_token_pool(embed_no_wm.last_hidden_state, batch_dict_no_wm["attention_mask"])
            
            ####### DETECTION ########

            for idx in range(len(decoded_output_no_wm)):
                if args.scheme != 'SynthID':
                    with torch.no_grad():
                        with autocast():
                            score_dict = watermark_detector.detect(decoded_output_w_wm[idx])
                        z_score = score_dict['z_score'].item()
                    
                    z_score_list.append(z_score) 
                else:
                    z_score = z_score_cur_batch[idx]
                
                STS_list.append(cos(embed_wm[idx], embed_no_wm[idx]).item())

                if args.log_generated_text:
                    dd = {
                        "prefix": data[i + idx],
                        "no_wm_completion": decoded_output_no_wm[idx], 
                        "gen_completion": decoded_output_w_wm[idx],
                        "z_wm": z_score,
                        "STS": cos(embed_wm[idx], embed_no_wm[idx]).item(),
                    }
                    with open(output_text_file, "a") as f:
                        f.write(json.dumps(dd) + "\n")
        
        end_time = time.time()
        total_time += (end_time - start_time)

        print("total_time", total_time)

        ######## OUTPUTS ########
        results = copy.deepcopy(vars(args))

        eval_results = {
            'z':{
                'avg': statistics.mean(z_score_list),
                'stdev': statistics.stdev(z_score_list),
                'total': len(z_score_list), 
            },
            'STS': statistics.mean(STS_list)
        }

        results.update(eval_results)
        if args.log_generated_text:
            with open(output_eval_file, "w") as outfile:
                outfile.write(json.dumps(results, indent=4))
        
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

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    main(args)