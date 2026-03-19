import torch
import numpy as np
import os, csv, zlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from difflib import SequenceMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fuzzy_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def main(args):
    # Quantization config to save VRAM for 8B models
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    print(f"Loading Target: {args.model_target}")
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.model_target, trust_remote_code=True)
    if not tgt_tokenizer.pad_token: tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
    
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.model_target, 
        quantization_config=bnb_config, # Comment this out if you have >40GB VRAM
        device_map="auto", 
        trust_remote_code=True
    )

    print(f"Loading Reference: {args.model_ref}")
    ref_tokenizer = AutoTokenizer.from_pretrained(args.model_ref, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_ref, 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )

    # --- Data Handling ---
    # Convert list of documents into one large searchable stream
    ds = " ".join(args.corpus_data)
    
    results_data = []
    seq_len, input_len = 50, 150 
    num_batches = int(np.ceil(args.N / args.batch_size))
    
    print(f"Attacking {args.N} samples...")
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            prompts, suffixes = [], []
            while len(prompts) < args.batch_size:
                r = np.random.randint(0, len(ds) - 3000)
                chunk = ds[r:r+3000] 
                tokens = tgt_tokenizer(chunk, return_tensors="pt")['input_ids'][0]
                if len(tokens) < (input_len + seq_len): continue
                    
                prompts.append(tgt_tokenizer.decode(tokens[:input_len]))
                suffixes.append(tgt_tokenizer.decode(tokens[input_len:input_len+seq_len]))

            inputs = tgt_tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = tgt_model.generate(**inputs, max_new_tokens=seq_len, do_sample=False, pad_token_id=tgt_tokenizer.eos_token_id)

            for i, out_seq in enumerate(outputs):
                full_text = tgt_tokenizer.decode(out_seq, skip_special_tokens=True)
                gen_suffix = full_text[len(prompts[i]):] 
                
                ppl_tgt = calculate_perplexity(full_text, tgt_model, tgt_tokenizer)
                ppl_ref = calculate_perplexity(full_text, ref_model, ref_tokenizer)
                ppl_ratio = ppl_ref / ppl_tgt if ppl_tgt > 0 else 0
                
                fuzzy = get_fuzzy_score(suffixes[i].strip(), gen_suffix.strip())
                is_exact = 1 if fuzzy > 0.95 else 0
                is_mosaic = 1 if (ppl_ratio > 2.0 and is_exact == 0 and fuzzy > 0.45) else 0
                
                results_data.append({
                    "is_exact": is_exact, "is_mosaic": is_mosaic,
                    "ppl_ratio": ppl_ratio, "fuzzy_score": fuzzy,
                    "zlib": len(zlib.compress(bytes(full_text, 'utf-8')))
                })
            pbar.update(args.batch_size)

    # Save logic
    model_tag = args.model_target.split('/')[-1]
    output_csv = f"results_{model_tag}_{args.name_tag}.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
        writer.writeheader()
        writer.writerows(results_data)
    print(f"Done! Results in {output_csv}")
