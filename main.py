import torch
from datasets import load_dataset
import numpy as np
from pprint import pprint, pformat
import logging
from helper import parse_lang, print_best, calculate_perplexity
import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
#for pythia
from transformers import GPTNeoXForCausalLM, AutoTokenizer
#for gptneo
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset
import csv
import os
import itertools
from types import SimpleNamespace

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model1)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model1 = GPTNeoXForCausalLM.from_pretrained(args.model1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model2 = GPTNeoXForCausalLM.from_pretrained(args.model2, return_dict=True).to(device)
    model2.eval()

    samples, prompts_list, prompt_suffix = [], [], []
    scores = {"mem":[], "XL": [], "S": [], "Lower": [], "zlib": []}
    seq_len, input_len = 256, 150
    num_batches = int(np.ceil(args.N / args.batch_size))
    #change this to parse_pilecorpus() for eng runs
    ds = parse_lang(args.corpus_path)
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            input_ids, attention_mask = [], []

            while len(input_ids) < args.batch_size:
                r = np.random.randint(0, len(ds))
                prompt = " ".join(ds[r:r+10000].split(" ")[1:-1])
                prompt_suff = prompt
                inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                prompt_suffix.append(prompt_suff)
                if len(inputs['input_ids'][0]) == input_len:
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])

            inputs = {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_mask)
            }

            prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            print("Attention Mask shape:", inputs['attention_mask'].shape)

            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True,
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            prompts_list.append(prompts)

            for text in texts:
                p1 = calculate_perplexity(text, model1, tokenizer)
                p2 = calculate_perplexity(text, model2, tokenizer)
                p_lower = calculate_perplexity(text.lower(), model1, tokenizer)
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["XL"].append(p1.cpu().item() if torch.is_tensor(p1) else p1)
                scores["S"].append(p2.cpu().item() if torch.is_tensor(p2) else p2)
                scores["Lower"].append(p_lower.cpu().item() if torch.is_tensor(p_lower) else p_lower)
                scores["zlib"].append(zlib_entropy)
            pbar.update(args.batch_size)

    scores = {k: np.asarray(v) for k, v in scores.items()}
    model1_name = args.model1.replace("/", "_")
    model2_name = args.model2.replace("/", "_")
    sample_test = [s[:200] for s in samples]
    comparison_result = [1 if sample == prompt else 0 for sample, prompt in zip(sample_test, prompt_suffix)]
    memorization = (sum(comparison_result) / len(comparison_result)) * 100
    prompts_list = [item for sublist in prompts_list for item in sublist]

    print("Memorization is:", memorization)

    output_csv = f'output_scores_{model1_name}_{model2_name}_{args.name_tag}.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['sample', 'prompt', 'suffix', 'memorized', 'PPL_XL', 'PPL_S', 'PPL_Lower', 'Zlib'])
        writer.writeheader()
        for sample, prompt, suff, mem, xl, s, lower, zlib_ in zip(samples, prompts_list, prompt_suffix, comparison_result, scores["XL"], scores["S"], scores["Lower"], scores["zlib"]):
            writer.writerow({
                'sample': sample, 'prompt': prompt, 'suffix': suff, 'memorized': mem,
                'PPL_XL': xl, 'PPL_S': s, 'PPL_Lower': lower, 'Zlib': zlib_
            })

    print("Results saved to", output_csv)

    output_txt = f'output_results_{model1_name}_{model2_name}_{args.name_tag}.txt'
    with open(output_txt, 'w') as f:
        metric = -np.log(scores["XL"])
        f.write("======== top sample by XL perplexity: ========\n")
        f.write(print_best(metric, samples, "PPL", scores["XL"]))
        f.write("\n")

        metric = np.log(scores["S"]) / np.log(scores["XL"])
        f.write("======== top sample by ratio of S and XL perplexities: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"]))
        f.write("\n")

        metric = np.log(scores["Lower"]) / np.log(scores["XL"])
        f.write("======== top sample by ratio of lower-case and normal-case perplexities: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"]))
        f.write("\n")

        metric = scores["zlib"] / np.log(scores["XL"])
        f.write("======== top sample by ratio of Zlib entropy and XL perplexity: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"]))
        f.write(f"======== Percentage of memorization is: ========\n{memorization}")

    print("Top results written to", output_txt)


