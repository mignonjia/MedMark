import torch
# import sys
# sys.path.append("/home/mingjia/med_watermark/synthid-text/src")


def generate_responses(model, tokenizer, inputs, gen_params, logits_processor, CONFIG, consider_ngram=False):
    device = model.device
    _, inputs_len = inputs['input_ids'].shape
    H = CONFIG['ngram_len'] - 1 # context window

    outputs_full = model.generate(
        **inputs,
        **gen_params
    )

    outputs = outputs_full[:, inputs_len:]
    # decoded_outputs = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

    # if consider_ngram==True, then only set the mask to be the non-repeated ngrams
    if consider_ngram:
        # eos mask is computed, skip first ngram_len - 1 tokens
        # eos_mask will be of shape [batch_size, output_len]
        eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=tokenizer.eos_token_id,
        )[:, H :]

        # context repetition mask is computed
        context_repetition_mask = logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )
        # context repitition mask shape [batch_size, output_len - (ngram_len - 1)]

        combined_mask = context_repetition_mask * eos_token_mask

    else:
        combined_mask = torch.ones((outputs.shape[0], outputs.shape[1] - H), dtype=torch.bool).to(device)

    g_values = logits_processor.compute_g_values(
        input_ids=outputs,
    )
    # g values shape [batch_size, output_len - (ngram_len - 1), depth]
    return g_values, combined_mask, outputs_full


def generate_g_values_and_mask(text, tokenizer, device, logits_processor, CONFIG, consider_ngram=False):
    outputs = tokenizer(text, return_tensors='pt')["input_ids"].to(device)
    outputs = torch.stack([
        row[1:] if row[0] == tokenizer.bos_token_id else row
        for row in outputs
    ])

    H = CONFIG['ngram_len'] - 1 # context window

    if consider_ngram:
        # eos mask is computed, skip first ngram_len - 1 tokens
        # eos_mask will be of shape [batch_size, output_len]
        eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=tokenizer.eos_token_id,
        )[:, H :]

        # context repetition mask is computed
        context_repetition_mask = logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )
        # context repitition mask shape [batch_size, output_len - (ngram_len - 1)]

        combined_mask = context_repetition_mask * eos_token_mask

    else:
        combined_mask = torch.ones((outputs.shape[0], outputs.shape[1] - H), dtype=torch.bool).to(device)

    g_values = logits_processor.compute_g_values(
        input_ids=outputs,
    )
    # g values shape [batch_size, output_len - (ngram_len - 1), depth]
    return g_values, combined_mask