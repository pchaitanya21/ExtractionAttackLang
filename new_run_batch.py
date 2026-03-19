import os
import gc 
import torch
from types import SimpleNamespace
from datasets import load_dataset
from new_main import main 

def get_training_samples(lang, n_samples=1000):
    """Streams actual training data for the specialized models."""
    samples = []
    if lang == "swahili":
        print(">>> Streaming Swahili data from lelapa/Inkuba-Mono...")
        ds = load_dataset("lelapa/Inkuba-Mono", split="train", streaming=True)
    elif lang == "finnish":
        print(">>> Streaming Finnish data from FineWeb-2 (subset: fi)...")
        ds = load_dataset("HuggingFaceFW/fineweb-2", "fi", split="train", streaming=True)
    
    # Collect samples that are long enough to be useful
    for entry in ds:
        text = entry['text']
        if len(text.split()) > 250: # Ensure document is long enough for prompt+suffix
            samples.append(text)
        if len(samples) >= n_samples:
            break
    return samples

def run_phase3_specialized():
    # Defined Experiment Pairs: (Target, Reference, Language)
    experiments = [
        ("lelapa/InkubaLM-0.4B", "EleutherAI/pythia-410m", "swahili"),
        ("LumiOpen/Llama-Poro-2-8B-base", "meta-llama/Llama-3.1-8B", "finnish")
    ]

    for tgt, ref, lang in experiments:
        print(f"\n{'='*60}")
        print(f"TARGET: {tgt} | REF: {ref} | LANG: {lang.upper()}")
        print(f"{'='*60}")

        # 1. Fetch actual training data
        raw_data = get_training_samples(lang, n_samples=500) # Lowered for testing speed

        args = SimpleNamespace(
            N=len(raw_data), 
            batch_size=2, # Keep low for 8B models to avoid VRAM OOM
            model_target=tgt,
            model_ref=ref,
            corpus_data=raw_data, # Passing data directly
            name_tag=f"phase3_native_{lang}"
        )

        try:
            main(args)
        except Exception as e:
            print(f"Error running {tgt}: {e}")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_phase3_specialized()
