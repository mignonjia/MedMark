import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import LlamaForCausalLM, AwqConfig
import copy 
import sys
sys.path.append("/home/mingjia/med_watermark/synthid-text/src")
from synthid_text import synthid_mixin


def model_loader_dist_shift(args, device, detect_only=False):
    d_type = torch.float16 if args.load_fp16 else torch.float

    if 'mistral' in args.model_name_or_path.lower():
        model_short_name = 'mistral'
        if not detect_only:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=d_type).to(device)
            embed_matrix = model.get_input_embeddings().weight
        else:
            embed_matrix = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=d_type).get_input_embeddings().weight.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer_trained = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        if not detect_only:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        embed_matrix = torch.load("data/embed/embedding_matrix_fp16.pt", map_location=device, weights_only=True)
        tokenizer_trained = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
        tokenizer_trained.pad_token_id = tokenizer_trained.eos_token_id
        if 'llama' in args.model_name_or_path.lower():
            model_short_name = 'llama'
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
            if "70b" not in args.model_name_or_path.lower():
                model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map=device, torch_dtype=d_type)
            else:
                # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                # model = AutoModelForCausalLM.from_pretrained(
                #     args.model_name_or_path,
                #     device_map=device, 
                #     quantization_config=quantization_config  
                # )
                quantization_config = AwqConfig(
                    bits=4,
                    fuse_max_seq_len=512, # Note: Update this as per your use-case
                    do_fuse=True
                )
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=d_type,
                    low_cpu_mem_usage=True,
                    device_map=device,
                    quantization_config=quantization_config
                )

            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if not detect_only:
        model.eval()
        for _, param in model.named_parameters():
            param.requires_grad = False
        embed_matrix.requires_grad = False

        terminators = [tokenizer.eos_token_id]
        if 'llama-3.1' in args.model_name_or_path.lower():
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        return model_short_name, model, tokenizer, tokenizer_trained, embed_matrix, terminators
    else:
        return model_short_name, None, tokenizer, tokenizer_trained, embed_matrix, None
    

def get_synthid_config(args, device):
    key_full_list = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960]
    config = dict({
        "ngram_len": args.H + 1,  # This corresponds to H=1 context window size in the paper.
        "keys": key_full_list[:args.layer],
        "num_leaves": args.num_leaves,
        "sampling_table_size": 2**16,
        "sampling_table_seed": 0,
        "context_history_size": 1024,
        "device": (device),
    })
    return config


### When using this function, the wm config is the default setting in synthid_mixin.py
def model_loader_SynthID(args, device):
    d_type = torch.float16 if args.load_fp16 else torch.float

    if 'mistral' in args.model_name_or_path.lower():
        model_short_name = 'mistral'
    CONFIG = synthid_mixin.DEFAULT_WATERMARKING_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_wm = synthid_mixin.SynthIDMistralForCausalLM.from_pretrained(args.model_name_or_path, device_map=device, torch_dtype=d_type)
    model_uwm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=device, torch_dtype=d_type)
    for m in [model_wm, model_uwm]:
        m.generation_config.pad_token_id = m.generation_config.eos_token_id
        m.eval()
        for _, param in m.named_parameters():
            param.requires_grad = False

    terminators = [tokenizer.eos_token_id]
    if 'llama-3.1' in args.model_name_or_path.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    return model_short_name, model_wm, model_uwm, tokenizer, CONFIG, terminators


def tokenizer_loader_SynthID(args):
    if 'mistral' in args.model_name_or_path.lower():
        model_short_name = 'mistral'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model_short_name, tokenizer


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def model_loader_STS(args, device_STS):
    d_type = torch.float16 if args.load_fp16 else torch.float
    model_STS = AutoModel.from_pretrained(args.model_STS, device_map=device_STS, torch_dtype=d_type)# .to(device_STS)
    tokenizer_STS = AutoTokenizer.from_pretrained(args.model_STS)
    tokenizer_STS.pad_token_id = tokenizer_STS.eos_token_id

    model_STS.eval()
    for name, param in model_STS.named_parameters():
        param.requires_grad = False

    return model_STS, tokenizer_STS


def get_gen_params(args, terminators, human=False):
    
    # By setting max_new_tokens = min_new_tokens = args.new_tokens, we control the output of samples to always be new_tokens
    if human:
        gen_params = dict(
            do_sample=args.use_sampling,
            max_new_tokens=args.new_tokens,
            eos_token_id=terminators
        )
    else:
        gen_params = dict(
            do_sample=args.use_sampling,
            min_new_tokens=args.new_tokens,
            max_new_tokens=args.new_tokens,
            eos_token_id=terminators
        )

    # Add sampling-specific parameters if sampling is enabled
    if args.use_sampling:
        gen_params.update(
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.sampling_temp
        )

    if args.no_repeat_ngram_size != "None":
        gen_params.update(
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )
    return gen_params