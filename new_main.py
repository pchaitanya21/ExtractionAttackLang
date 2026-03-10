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
    """Calculates lexical similarity for Mosaic Memory detection (0.0 to 1.0)"""
    return SequenceMatcher(None, a, b).ratio()

def calculate_perplexity(text, model, tokenizer):
    """Calculates perplexity using the specific model's tokenizer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def get_model_embedding(model, tokenizer, text):
    """Extracts Language Embedding for Topological Graph"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

def main(args):
    print(f"Loading TARGET model: {args.model_target}...")
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.model_target, trust_remote_code=True)
    tgt_tokenizer.padding_side = "left"
    if not tgt_tokenizer.pad_token: tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
    
    tgt_model = AutoModelForCausalLM.from_pretrained(args.model_target, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    print(f"Loading REFERENCE model: {args.model_ref}...")
    # CRITICAL: Load Ref Tokenizer to prevent cross-architecture crashes
    ref_tokenizer = AutoTokenizer.from_pretrained(args.model_ref, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_ref, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    # Note: Replace this with your actual parsing logic
    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        ds = f.read()
    
    results_data = []
    seq_len, input_len = 50, 150 
    num_batches = int(np.ceil(args.N / args.batch_size))
    
    print(f"Starting Mosaic Attack on {args.N} samples...")
    
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            prompts, suffixes = [], []
            
            while len(prompts) < args.batch_size:
                r = np.random.randint(0, len(ds) - 2000)
                chunk = ds[r:r+2000] 
                tokenized = tgt_tokenizer(chunk, return_tensors="pt")['input_ids'][0]
                if len(tokenized) < (input_len + seq_len): continue
                    
                prompts.append(tgt_tokenizer.decode(tokenized[:input_len], skip_special_tokens=True))
                suffixes.append(tgt_tokenizer.decode(tokenized[input_len:input_len+seq_len], skip_special_tokens=True))

            inputs = tgt_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                output_sequences = tgt_model.generate(**inputs, max_new_tokens=seq_len, do_sample=False, pad_token_id=tgt_tokenizer.eos_token_id)

            for i, out_seq in enumerate(output_sequences):
                full_text = tgt_tokenizer.decode(out_seq, skip_special_tokens=True)
                generated_suffix = full_text[len(prompts[i]):] 
                
                # Metrics
                ppl_tgt = calculate_perplexity(full_text, tgt_model, tgt_tokenizer)
                ppl_ref = calculate_perplexity(full_text, ref_model, ref_tokenizer) # Use Ref Tokenizer here
                ppl_ratio = ppl_ref / ppl_tgt if ppl_tgt > 0 else 0
                
                fuzzy_score = get_fuzzy_score(suffixes[i].strip(), generated_suffix.strip())
                is_exact = 1 if fuzzy_score > 0.95 else 0 
                
                # THE MOSAIC SIGNAL: High Ratio (Discovered) but Low Exactness (Not Recovered)
                is_mosaic = 1 if (ppl_ratio > 2.0 and is_exact == 0 and fuzzy_score > 0.5) else 0
                
                zlib_score = len(zlib.compress(bytes(full_text, 'utf-8')))
                lang_emb = get_model_embedding(tgt_model, tgt_tokenizer, full_text)
                
                results_data.append({
                    "prompt": prompts[i],
                    "gold_suffix": suffixes[i],
                    "gen_suffix": generated_suffix,
                    "is_exact": is_exact,
                    "fuzzy_score": fuzzy_score,
                    "is_mosaic": is_mosaic,
                    "ppl_tgt": ppl_tgt,
                    "ppl_ref": ppl_ref,
                    "ppl_ratio": ppl_ratio,
                    "zlib": zlib_score,
                    "embedding_mean": np.mean(lang_emb) 
                })
            pbar.update(args.batch_size)

    model_tag = f"{args.model_target.replace('/', '_')}_vs_{args.model_ref.replace('/', '_')}"
    output_csv = f"results_{model_tag}_{args.name_tag}.csv"
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
        writer.writeheader()
        writer.writerows(results_data)
    print(f"Saved extended metrics to {output_csv}")
