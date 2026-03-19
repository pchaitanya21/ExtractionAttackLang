import gc
import torch
from types import SimpleNamespace
from datasets import load_dataset
from new_main import main 

def get_training_samples(lang, n_samples=1000):
    """Streams actual training data for Phase 3 models."""
    samples = []
    if lang == "swahili":
        print(f">>> Streaming Swahili data from lelapa/Inkuba-Mono...")
        ds = load_dataset("lelapa/Inkuba-Mono", split="train", streaming=True)
    elif lang == "finnish":
        print(f">>> Streaming Finnish data from FineWeb-2 (subset: fi)...")
        ds = load_dataset("HuggingFaceFW/fineweb-2", "fi", split="train", streaming=True)
    
    for entry in ds:
        text = entry['text']
        # Ensure the text is long enough for a 150-token prompt + 50-token suffix
        if len(text.split()) > 250:
            samples.append(text)
        if len(samples) >= n_samples:
            break
    return samples

def run_specialized_phase():
    # Define (Target Model, Reference Model, Language)
    experiments = [
        ("lelapa/InkubaLM-0.4B", "EleutherAI/pythia-410m", "swahili"),
        ("LumiOpen/Llama-Poro-2-8B-base", "meta-llama/Llama-3.1-8B", "finnish")
    ]

    for tgt, ref, lang in experiments:
        print(f"\n{'='*50}\nRUNNING: {lang.upper()}\nTarget: {tgt}\nRef: {ref}\n{'='*50}")

        # 1. Fetch data from the actual source
        raw_data = get_training_samples(lang, n_samples=1000)

        args = SimpleNamespace(
            N=1000, 
            batch_size=2, # Keep low for 8B models
            model_target=tgt,
            model_ref=ref,
            corpus_data=raw_data, # Data passed as list
            name_tag=f"phase3_{lang}"
        )

        try:
            main(args)
        except Exception as e:
            print(f"❌ Error in {lang} experiment: {e}")
        
        # VRAM Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
