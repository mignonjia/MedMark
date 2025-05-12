import os
import argparse
# os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")
import json

from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
import random
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.amp import autocast, GradScaler
import statistics
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.checkpoint

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.MedMark_networks import GammaNetwork, DeltaNetwork, GammaLSTM, DeltaLSTM

# Reproducible
hash_key = 15485863
torch.manual_seed(hash_key)
torch.cuda.manual_seed(hash_key)

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def add_tokens(input_ids):
    # Add 1 at the beginning and 2 at the end for each sequence
    start_token = torch.tensor([1], dtype=input_ids.dtype, device=input_ids.device)
    end_token = torch.tensor([2], dtype=input_ids.dtype, device=input_ids.device)
    
    modified_input_ids = []
    for sequence in input_ids:
        new_sequence = torch.cat([start_token, sequence, end_token])
        modified_input_ids.append(new_sequence)
    
    return torch.stack(modified_input_ids)

def get_green_mask(scores, gamma, tau):
    # Use Gumbel Softmax to Generate Green/Red Tokens
     
    device = scores.device
    green_tokens_mask = torch.zeros_like(scores, dtype=torch.float)
    red_tokens_mask = torch.zeros_like(scores, dtype=torch.float)
   
    logits_bernoulli = torch.zeros(scores.shape[0], scores.shape[1], 2)
    
    logits_bernoulli[:,:,0] = (1-gamma).expand_as(logits_bernoulli[:,:,0])
    logits_bernoulli[:,:,1] = gamma.expand_as(logits_bernoulli[:,:,1])

    logits_bernoulli = torch.log(logits_bernoulli) # [batch_size, vocab_size, 2]
    red_green_prob = F.gumbel_softmax(logits_bernoulli, tau=tau, hard=False, dim=-1).to(device)  # [batch_size, vocab_size, 2]

    for b_id in range(scores.shape[0]):
        green_tokens_mask[b_id] = red_green_prob[b_id,:,1].squeeze(0) # [vocab_size]
        red_tokens_mask[b_id] = red_green_prob[b_id,:,0].squeeze(0)

    return green_tokens_mask, red_tokens_mask # [batch_size, vocab_size]

def differentiable_decode(model, model_STS, delta_network, gamma_network, input_ids, attention_masks, max_length, \
                          scale=20, tau=0.1, fix_gamma=False, init_gamma=None):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    delta_list = torch.empty((batch_size, 0), dtype=torch.float).to(device)
    gamma_list = torch.empty((batch_size, 0), dtype=torch.float).to(device)
    p_green_list = torch.empty((batch_size, 0), dtype=torch.float).to(device)

    embedding_matrix = model.get_input_embeddings().weight
    embedding_matrix_STS = model_STS.get_input_embeddings().weight

    bos_embedding = model_STS.get_input_embeddings()(torch.tensor([1], device=device)).squeeze().expand(batch_size, 1, -1)
    eos_embedding = model_STS.get_input_embeddings()(torch.tensor([2], device=device)).squeeze().expand(batch_size, 1, -1)

    input_embeddings_wm = model.get_input_embeddings()(input_ids).to(device)
    # input_embeddings_wm_STS = model_STS.get_input_embeddings()(input_ids).to(device)
    input_embeddings_wm_STS = bos_embedding.clone()
    input_embeddings_no_wm = model.get_input_embeddings()(input_ids).to(device)
    # input_embeddings_no_wm_STS = model_STS.get_input_embeddings()(input_ids).to(device)
    input_embeddings_no_wm_STS = bos_embedding.clone()

    output_ids_no_wm = torch.empty((batch_size, 0), dtype=torch.int).to(device)
    output_ids_no_wm = torch.cat([input_ids, output_ids_no_wm], dim=1)

    output_ids_wm = torch.empty((batch_size, 0), dtype=torch.int).to(device)
    output_ids_wm = torch.cat([input_ids, output_ids_wm], dim=1)

    if isinstance(delta_network, DeltaNetwork):
        adaptor = "MLP"
    else:
        adaptor = "LSTM"

    for _ in range(max_length):
        ########### Generate wm embeddings ###########
        with torch.no_grad():
            logits = model(inputs_embeds=input_embeddings_wm, attention_mask=attention_masks).logits
            
        # Get llm logits 
        last_logits = logits[:, -1, :].squeeze(dim=1)

        # Get delta/gamma based on embedding of preceding token
        if adaptor == "MLP":
            delta = delta_network(input_embeddings_wm[:, -1, :].squeeze(dim=1) * scale) 
            if fix_gamma:
                gamma = torch.Tensor([init_gamma]).to(device)
            else:
                gamma = gamma_network(input_embeddings_wm[:, -1, :].squeeze(dim=1) * scale) 
        else:
            delta = delta_network(input_embeddings_wm[:, -5:, :] * scale) 
            if fix_gamma:
                gamma = torch.Tensor([init_gamma]).to(device)
            else:
                gamma = gamma_network(input_embeddings_wm[:, -5:, :] * scale) 

        # print("input_embeddings_wm", input_embeddings_wm[:, -1, :])
        # for name, param in delta_network.named_parameters():
        #     print(name, param) 

        # Get green tokens using Gumbel Softmax
        green_mask, red_mask = get_green_mask(last_logits, gamma, tau) # [batch_size, vocab_size]
        last_logits = last_logits + delta * green_mask # [batch_size, vocab_size] both for last_logits and (delta * green_mask)

        # # Option 1: compute soft_one_hot (contains gradient) and hard_one_hot (no gradient), and get mix_ont_hot 
        # soft_one_hot = F.softmax(last_logits, dim=-1).to(embedding_matrix.dtype)
        # max_val, max_index = torch.max(soft_one_hot, dim=-1, keepdim=True)
        # hard_one_hot = torch.zeros_like(soft_one_hot)
        # hard_one_hot[soft_one_hot == max_val] = 1 #[batch_size]
        # mix_one_hot = hard_one_hot * 0.5 + soft_one_hot * 0.5

        # Option 2: compute gumbel_one_hot
        gumbel_one_hot = F.gumbel_softmax(last_logits, tau=tau, hard=False, dim=-1).to(embedding_matrix.dtype)
        max_val, max_index = torch.max(gumbel_one_hot, dim=-1, keepdim=True)
        mix_one_hot = gumbel_one_hot

        output_ids_wm = torch.cat([output_ids_wm, max_index], dim=1)
        
        # Get next token embedding
        predicted_embedding_wm = torch.matmul(mix_one_hot, embedding_matrix)
        input_embeddings_wm = torch.cat([input_embeddings_wm, predicted_embedding_wm.unsqueeze(1)], dim=1)

        predicted_embedding_STS = torch.matmul(mix_one_hot, embedding_matrix_STS)
        input_embeddings_wm_STS = torch.cat([input_embeddings_wm_STS, predicted_embedding_STS.unsqueeze(1)], dim=1)
        # print(predicted_embedding_STS)

        # Probability of green token to compute detection loss
        p_green = torch.sum(mix_one_hot * green_mask, dim=-1, keepdim=True)
   
        delta_list = torch.cat([delta_list, delta.expand(batch_size, 1)], dim=-1)
        gamma_list = torch.cat([gamma_list, gamma.expand(batch_size, 1)], dim=-1)

        p_green_list = torch.cat([p_green_list, p_green], dim=-1)

        ########### Generate no wm embeddings ###########
        with torch.no_grad():
            logits_no_wm = model(inputs_embeds=input_embeddings_no_wm, attention_mask=attention_masks).logits

        last_logits_no_wm = logits_no_wm[:, -1, :].squeeze(dim=1)

        # Get soft_one_hot_no_wm and hard_one_hot_no_wm, and compute mix_ont_hot_no_wm. 
        # None of these contain gradient. 
        # This is just to make sure the generating no_wm embeddings follows the same procedure as the above wm embeddings.

        # # Option 1: compute soft_one_hot (contains gradient) and hard_one_hot (no gradient), and get mix_ont_hot 
        # soft_one_hot_no_wm = F.softmax(last_logits_no_wm, dim=-1).to(embedding_matrix.dtype)
        # max_val, max_index_no_wm = torch.max(soft_one_hot_no_wm, dim=-1, keepdim=True)
        # hard_one_hot_no_wm = torch.zeros_like(soft_one_hot_no_wm)
        # hard_one_hot_no_wm[soft_one_hot_no_wm == max_val] = 1 
        # mix_one_hot_no_wm = hard_one_hot_no_wm * 0.5 + soft_one_hot_no_wm * 0.5

        # Option 2: compute gumbel_one_hot
        gumbel_one_hot_no_wm = F.gumbel_softmax(last_logits_no_wm, tau=tau, hard=False, dim=-1).to(embedding_matrix.dtype)
        max_val, max_index_no_wm = torch.max(gumbel_one_hot_no_wm, dim=-1, keepdim=True)
        mix_one_hot_no_wm = gumbel_one_hot_no_wm

        # Embedding of the next token
        predicted_embedding_no_wm = torch.matmul(mix_one_hot_no_wm, embedding_matrix)
        input_embeddings_no_wm = torch.cat([input_embeddings_no_wm, predicted_embedding_no_wm.unsqueeze(1)], dim=1)
        
        predicted_embedding_no_wm_STS = torch.matmul(mix_one_hot_no_wm, embedding_matrix_STS)
        input_embeddings_no_wm_STS = torch.cat([input_embeddings_no_wm_STS, predicted_embedding_no_wm_STS.unsqueeze(1)], dim=1) 
        
        output_ids_no_wm = torch.cat([output_ids_no_wm, max_index_no_wm], dim=1)

        attention_masks = torch.cat((attention_masks, torch.ones(batch_size, 1).cuda()), dim=1)

    input_embeddings_wm_STS = torch.cat([input_embeddings_wm_STS, eos_embedding.clone()], dim=1)
    input_embeddings_no_wm_STS = torch.cat([input_embeddings_no_wm_STS, eos_embedding.clone()], dim=1)

    # print(input_embeddings_wm)

    return input_embeddings_wm_STS, output_ids_wm, input_embeddings_no_wm_STS, output_ids_no_wm, delta_list, gamma_list, p_green_list


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )

    scaler = GradScaler("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype).to(device)
    hidden_size = model.config.hidden_size
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model_STS = AutoModel.from_pretrained(args.model_STS_name_or_path, torch_dtype=dtype).to(device)
    model_STS.eval()

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model_STS.named_parameters():
        param.requires_grad = False
    
    if args.adaptor == 'MLP':
        delta_network = DeltaNetwork(input_dim=hidden_size, layers=args.layer_delta, init_val=args.init_val_delta).to(device)
        gamma_network = GammaNetwork(input_dim=hidden_size, layers=args.layer_gamma, init_val=args.init_val_gamma).to(device)
    elif args.adaptor == 'LSTM':
        delta_network = DeltaLSTM(input_dim=hidden_size, init_val=args.init_val_delta, device=device)
        gamma_network = GammaLSTM(input_dim=hidden_size, init_val=args.init_val_gamma, device=device)
    else:
        print("Adpator network doesn't exist!")
        return 
    
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(list(gamma_network.parameters()) + list(delta_network.parameters()), lr=args.lr, weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.Adam(list(gamma_network.parameters()) + list(delta_network.parameters()), lr=args.lr, weight_decay=args.wdecay)
        
    scheduler_gamma = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.lr_step_size, gamma = args.lr_scheduler)

    ###########################################################################
    # Compose the partials to create the pipeline
    ###########################################################################

    def data_collator(input_ids):
        # Reverse the sequences, pad them, and then reverse back
        input_ids_reversed = [torch.flip(tensor, [0]) for tensor in input_ids]  # Reverse each sequence
        input_ids_padded_reversed = pad_sequence(input_ids_reversed, batch_first=True, padding_value=tokenizer.pad_token_id)
        input_ids_padded = torch.flip(input_ids_padded_reversed, [1])  # Reverse back to original order
        return input_ids_padded
    
    with open("data/HealthSearchQA/QA_train.json", "r") as f:
        entrys = json.load(f)
    with open(f"data/mistral_template/qa_template.txt") as f:
        lines = f.readlines()
        prompt_template = "".join(lines)
    entrys = [prompt_template.replace('TEMPLATE', val.strip()) for val in entrys]

    pbar = tqdm(total=args.min_generations)
    
    # Record all the L_S and L_D during training for analysis
    L_S_list, L_D_list = [], [] # empty after 1 epoch
    L_S_list_100, L_D_list_100 = [], [] # empty after 100 steps
    step_cnter = 0
    z_score_factor = float(args.z_score_factor) # change string to float. string is used during output


    for epoch in range(args.epochs):

        random.shuffle(entrys)  # <- shuffle the data at the start of each epoch
           
        for i in range(0, len(entrys), args.batch_size):
            cur_data = entrys[i:(i+args.batch_size)]

            input_ids_list = []

            # Encode each entry one by one and collect the input_ids
            for entry in cur_data:
                encoded_entry = tokenizer(entry, max_length=args.prompt_tokens, return_tensors="pt")
                input_ids_list.append(encoded_entry['input_ids'].view(-1))
            
            input_ids = data_collator(input_ids_list).to(model.device)
                   
            attention_masks = (input_ids != tokenizer.pad_token_id).long()

            optimizer.zero_grad()

            with autocast("cuda"):
                # Get embedding of wm_gen (with gradient) and no_wm_gen (no gradient)
                ret = differentiable_decode(model, model_STS, delta_network, gamma_network, input_ids, attention_masks, args.max_new_tokens, \
                                                scale=args.scale, tau=args.tau, fix_gamma=args.fix_gamma, init_gamma=args.init_val_gamma)
                input_embeddings_wm_STS, output_ids_wm, input_embeddings_no_wm_STS, output_ids_no_wm, delta_list, gamma_list, p_green_list = ret

                # STS embedding
                attention_masks = torch.ones(input_embeddings_wm_STS.shape[0], input_embeddings_wm_STS.shape[1]).to(device)
                embed_wm = model_STS(inputs_embeds=input_embeddings_wm_STS, attention_mask=attention_masks)
                embed_wm = last_token_pool(embed_wm.last_hidden_state, attention_masks)
                embed_no_wm = model_STS(inputs_embeds=input_embeddings_no_wm_STS, attention_mask=attention_masks)
                embed_no_wm = last_token_pool(embed_no_wm.last_hidden_state, attention_masks)
                
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

                # Get L_D (detection loss)
                var = torch.sum(gamma_list * (1 - gamma_list), dim = -1) # [batch_size]
                true_z_score = torch.sum(p_green_list - gamma_list, dim = -1)/torch.sqrt(var) # [batch_size]
                true_z_score = torch.mean(true_z_score)
                L_D_list_100.append(true_z_score.item())
                detect_loss = (torch.log10(torch.mean(true_z_score)) if args.log_z_score else torch.mean(true_z_score))
                detect_loss *= (- z_score_factor)

            STS_loss = - torch.mean(cos(embed_wm, embed_no_wm))
            L_S_list_100.append(-STS_loss.item())
            loss = STS_loss + detect_loss 

            # Step
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler_gamma.step()
            
            if args.wandb:
                log_dict = {
                    "semantic loss": -STS_loss,
                    "detect_loss": detect_loss,
                    "true_z_score": true_z_score,
                    "gamma_average": torch.mean(gamma_list),
                    "delta_average": torch.mean(delta_list)
                }
                run.log(log_dict, step=step_cnter) 
            step_cnter += 1
            
            if step_cnter % args.log_freq == 0: 
                
                # check the value range of delta/gamma by printing an example
                print("epoch", epoch)
                print("step", step_cnter)
                print("delta_list")
                print(delta_list[0]) 
                print("gamma_list")
                print(gamma_list[0]) 

                print("STS", statistics.mean(L_S_list_100))
                print("z-score", statistics.mean(L_D_list_100))

                L_S_list.append(statistics.mean(L_S_list_100))
                L_D_list.append(statistics.mean(L_D_list_100))

                L_S_list_100, L_D_list_100 = [], []

                if args.log_ckpt:
                    if args.fix_gamma:
                        checkpoint_path = f'{args.ckpt_dir}/Mistral/fix_{args.init_val_gamma}_{args.init_val_delta}/{args.z_score_factor}_{args.scale}_{args.lr}/{args.dtype}_{step_cnter}.pth'
                    else:
                        checkpoint_path = f'{args.ckpt_dir}/Mistral/gumbel_{args.init_val_gamma}_{args.init_val_delta}/{args.z_score_factor}_{args.scale}_{args.lr}/{args.dtype}_{step_cnter}.pth'
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                    checkpoint = {
                        'delta_state_dict': delta_network.state_dict(),
                        'gamma_state_dict': gamma_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        print("epoch", epoch)
        print("step", step_cnter)
        print("delta_list")
        print(delta_list[0]) 
        print("gamma_list")
        print(gamma_list[0]) 

        print("STS", statistics.mean(L_S_list_100))
        print("z-score", statistics.mean(L_D_list_100))

        L_S_list.append(statistics.mean(L_S_list_100))
        L_D_list.append(statistics.mean(L_D_list_100))

        L_S_list_100, L_D_list_100 = [], []

        if args.log_ckpt and epoch == 2:
            if args.fix_gamma:
                checkpoint_path = f'ckpt/Mistral/fix_{args.init_val_gamma}_{args.init_val_delta}/{args.z_score_factor}_{args.scale}_{args.lr}/{args.dtype}_{step_cnter}.pth'
            else:
                checkpoint_path = f'ckpt/Mistral/gumbel_{args.init_val_gamma}_{args.init_val_delta}/{args.z_score_factor}_{args.scale}_{args.lr}/{args.dtype}_{step_cnter}.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = {
                'delta_state_dict': delta_network.state_dict(),
                'gamma_state_dict': gamma_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
    
        print("=" * 30)
        print("Summrize for epoch ", epoch)
        print("Summrize for STS", statistics.mean(L_S_list))
        print("Summrize for z_score", statistics.mean(L_D_list))
        L_S_list, L_D_list = [], []
    pbar.close()

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_STS_name_or_path",
        type=str,
        default="Salesforce/SFR-Embedding-2_R",
        help="STS model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--log_ckpt",
        type=str2bool,
        default=True,
        help=("Whether to log ckpts"),
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="log frequency",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="Model data type, fp16 or bf16.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=20,
        help="scale Mistral input embedding",
    )
    parser.add_argument(
        "--adaptor",
        type=str,
        default="MLP",
        choices=["MLP", "LSTM"],
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--prompt_tokens",
        type=int,
        default=200,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use for generation.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
    )
    parser.add_argument(
        "--fix_gamma",
        type=str2bool,
        default=False,
        help="Whether fix gamma.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="YOUR_WANDB_PROJECT",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="YOUR_WANDB_ENTITY",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpt",
        help="ckpt directory.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for gamma-generator.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=50,
        help="Learning rate schedule step size.",
    )
    parser.add_argument(
        "--layer_gamma",
        type=int,
        default=2,
        help="number of layers for gamma network (MLP).",
    )
    parser.add_argument(
        "--layer_delta",
        type=int,
        default=2,
        help="number of layers for delta network (MLP).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=float,
        default=1.0,
        help="Learning rate schedule gamma (decay factor).",
    )
    parser.add_argument(
        "--init_val_gamma",
        type=float,
        default=0.25,
        help="init gamma.",
    )
    parser.add_argument(
        "--init_val_delta",
        type=float,
        default=2.0,
        help="init delta.",
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=1e-5,
        help="Weight decay.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Temperature for Gumbel Softmax.",
    )
    parser.add_argument(
        "--log_z_score",
        type=str2bool,
        default=False,
        help="computer log z score",
    )
    parser.add_argument(
        "--z_score_factor",
        type=str,
        default="1e-4",
        help="factor on detection loss compared with semantic loss",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="SGD or Adam",
    )
    args = parser.parse_args()

    main(args)
