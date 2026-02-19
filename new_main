import torch
import numpy as np
import os
import csv
import zlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper import parse_lang, print_best, calculate_perplexity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_embedding(model, tokenizer, text):
    """Extracts the average last-hidden-state (Language Embedding) for EMNLP graph analysis."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Average of the last layer's hidden states (Contextual Embedding)
    return outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

def main(args):
    # --- 1. Load Target Model (e.g., Pythia 2.8B) ---
    print(f"Loading TARGET model: {args.model_target}...")
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.model_target)
    tgt_tokenizer.padding_side = "left"
    if not tgt_tokenizer.pad_token:
        tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
    
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.model_target, 
        return_dict=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # --- 2. Load Reference Model (e.g., Pythia 70M) ---
    print(f"Loading REFERENCE model: {args.model_ref}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_ref, 
        return_dict=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    # Note: We use target tokenizer for ref model to ensure token alignment 
    # (Pythia/GPT-Neo families share tokenizers usually, but be careful if mixing families)

    # --- 3. Parse Data ---
    ds = parse_lang(args.corpus_path)
    
    samples = []
    results_data = []
    
    seq_len, input_len = 50, 150 # 150 prompt + 50 suffix
    num_batches = int(np.ceil(args.N / args.batch_size))
    
    print(f"Starting Attack on {args.N} samples...")
    
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            prompts = []
            suffixes = []
            
            # Random Sampling Logic
            while len(prompts) < args.batch_size:
                r = np.random.randint(0, len(ds) - 2000)
                # Simple extraction: grab a chunk
                chunk = ds[r:r+2000] 
                tokenized = tgt_tokenizer(chunk, return_tensors="pt")['input_ids'][0]
                
                if len(tokenized) < (input_len + seq_len):
                    continue
                    
                prompt_ids = tokenized[:input_len]
                suffix_ids = tokenized[input_len:input_len+seq_len]
                
                prompts.append(tgt_tokenizer.decode(prompt_ids, skip_special_tokens=True))
                suffixes.append(tgt_tokenizer.decode(suffix_ids, skip_special_tokens=True))

            # Batch Generation (Target Model Only)
            inputs = tgt_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                output_sequences = tgt_model.generate(
                    **inputs,
                    max_new_tokens=seq_len,
                    do_sample=False, # Greedy decoding is standard for extraction
                    pad_token_id=tgt_tokenizer.eos_token_id
                )

            # Analyze Results
            for i, out_seq in enumerate(output_sequences):
                full_text = tgt_tokenizer.decode(out_seq, skip_special_tokens=True)
                generated_suffix = full_text[len(prompts[i]):] # Approximate split
                
                # --- METRICS ---
                # 1. Raw Perplexity (Target)
                ppl_tgt = calculate_perplexity(full_text, tgt_model, tgt_tokenizer)
                
                # 2. Reference Perplexity (Reference)
                # Note: We must re-encode with target tokenizer if models differ, 
                # but for Pythia-Pythia pairs, it's safe.
                ppl_ref = calculate_perplexity(full_text, ref_model, tgt_tokenizer)
                
                # 3. Perplexity Ratio (MIA Signal)
                # Ratio > 1 implies Target is less surprised than Ref (Memorization)
                ppl_ratio = ppl_ref / ppl_tgt if ppl_tgt > 0 else 0
                
                # 4. Zlib Entropy
                zlib_score = len(zlib.compress(bytes(full_text, 'utf-8')))
                
                # 5. Language Embedding (for EMNLP Graph)
                # We extract this from the Target model's representation
                lang_emb = get_model_embedding(tgt_model, tgt_tokenizer, full_text)
                
                # 6. Exact Match Check
                is_memorized = 1 if suffixes[i].strip() == generated_suffix.strip() else 0
                
                results_data.append({
                    "prompt": prompts[i],
                    "gold_suffix": suffixes[i],
                    "gen_suffix": generated_suffix,
                    "memorized": is_memorized,
                    "ppl_tgt": ppl_tgt,
                    "ppl_ref": ppl_ref,
                    "ppl_ratio": ppl_ratio,
                    "zlib": zlib_score,
                    "embedding_mean": np.mean(lang_emb) # Saving mean to save CSV space
                })
            
            pbar.update(args.batch_size)

    # --- Save Results ---
    model_tag = f"{args.model_target.replace('/', '_')}_vs_{args.model_ref.replace('/', '_')}"
    output_csv = f"results_{model_tag}_{args.name_tag}.csv"
    
    keys = results_data[0].keys()
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_data)
        
    print(f"Saved extended metrics to {output_csv}")
