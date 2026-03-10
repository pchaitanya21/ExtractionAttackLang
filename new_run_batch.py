import os
import gc 
import torch
import itertools
from types import SimpleNamespace
from new_main import main 

def run_batch(corpus_paths):
    model_pairs = [
        # --- PHASE 1 & 2: The Control Models (The Pile) ---
        ("EleutherAI/pythia-2.8b", "EleutherAI/pythia-70m"),
        ("EleutherAI/pythia-1.4b", "EleutherAI/pythia-70m"),
        ("EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-125M"),
        
        # --- PHASE 3: Modern Generalization Models ---
        # 1. Swahili Specialist (Trained on Inkuba-Mono)
        ("lelapa/InkubaLM-0.4B", "EleutherAI/pythia-70m"),
        
        # 2. Finnish Specialist (Trained on 50B Finnish tokens)
        ("LumiOpen/Llama-Poro-2-8B-base", "EleutherAI/pythia-70m")
    ]

    for corpus_path, (tgt, ref) in itertools.product(corpus_paths, model_pairs):
        args = SimpleNamespace(
            N=2000, # Reduced N for speed, increase for final paper
            batch_size=10, # Lower batch size for dual-model loading
            model_target=tgt,
            model_ref=ref,
            corpus_path=corpus_path,
            name_tag=os.path.basename(corpus_path).replace('.txt', '')
        )

        print(f"\n>>> EXPERIMENT: {args.model_target} (Target) vs {args.model_ref} (Ref)")
        print(f">>> DATA: {args.corpus_path}")
        
        try:
            main(args)
        except Exception as e:
            print(f"Error running {tgt}: {e}")
        
        # Cleanup to prevent VRAM OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
