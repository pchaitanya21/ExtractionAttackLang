import torch
import numpy as np
import os
import csv
import zlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fuzzy_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def get_model_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

def main(args):
    # 1. Load Models
    print(f"Loading Target: {args.model_target}")
    # Note: Use load_in_4bit=True here if your GPU is under 24GB VRAM
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.model_target, trust_remote_code=True)
    tgt_model = AutoModelForCausalLM.from_pretrained(args.model_target, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    print(f"Loading Reference: {args.model_ref}")
    ref_tokenizer = AutoTokenizer.from_pretrained(args.model_ref, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_ref, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    # 2. Data Preparation
    # We join the documents into one stream or sample from the list
    ds_list = args.corpus_data
    results_data = []
    seq_len, input_len = 50, 150 
    
    print(f"Attacking {args.N} native samples...")
    
    with tqdm(total=args.N) as pbar:
        # Since we have a list of docs, we process them individually or in small batches
        for doc_idx in range(0, len(ds_list), args.batch_size):
            batch_docs = ds_list[doc_idx : doc_idx + args.batch_size]
            prompts, suffixes = [], []

            for doc in batch_docs:
                tokens = tgt_tokenizer(doc, return_tensors="pt")['input_ids'][0]
                if len(tokens) < (input_len + seq_len):
                    continue
                # Take a random slice from the document
                start_idx = np.random.randint(0, len(tokens) - (input_len + seq_len))
                prompts.append(tgt_tokenizer.decode(tokens[start_idx : start_idx + input_len]))
                suffixes.append(tgt_tokenizer.decode(tokens[start_idx + input_len : start_idx + input_len + seq_len]))

            if not prompts: continue

            inputs = tgt_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                output_sequences = tgt_model.generate(**inputs, max_new_tokens=seq_len, do_sample=False, pad_token_id=tgt_tokenizer.eos_token_id)

            for i, out_seq in enumerate(output_sequences):
                full_text = tgt_tokenizer.decode(out_seq, skip_special_tokens=True)
                gen_suffix = full_text[len(prompts[i]):]
                
                ppl_tgt = calculate_perplexity(full_text, tgt_model, tgt_tokenizer)
                ppl_ref = calculate_perplexity(full_text, ref_model, ref_tokenizer)
                ppl_ratio = ppl_ref / ppl_tgt if ppl_tgt > 0 else 0
                
                fuzzy = get_fuzzy_score(suffixes[i].strip(), gen_suffix.strip())
                is_exact = 1 if fuzzy > 0.95 else 0
                is_mosaic = 1 if (ppl_ratio > 2.0 and is_exact == 0 and fuzzy > 0.4) else 0

                results_data.append({
                    "is_exact": is_exact,
                    "is_mosaic": is_mosaic,
                    "ppl_ratio": ppl_ratio,
                    "fuzzy_score": fuzzy,
                    "gen_suffix": gen_suffix[:100], # truncated for CSV readability
                    "gold_suffix": suffixes[i][:100]
                })
            pbar.update(len(prompts))

    # 3. Save
    model_tag = args.model_target.split('/')[-1]
    output_csv = f"results_{model_tag}_{args.name_tag}.csv"
    keys = results_data[0].keys()
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_data)
    print(f"Results saved to {output_csv}")
