import os
import itertools
from types import SimpleNamespace
from main import main  # your main logic

def run_batch(corpus_paths):
    model_pairs = [
        ("EleutherAI/pythia-2.8b", "EleutherAI/pythia-1.4b"),
        ("EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-1.3B")
    ]

    for corpus_path, (model1, model2) in itertools.product(corpus_paths, model_pairs):
        args = SimpleNamespace(
            N=10000,
            batch_size=1000,
            model1=model1,
            model2=model2,
            corpus_path=corpus_path,
            name_tag=f"run_{model1.replace('/', '_')}_and_{model2.replace('/', '_')}_{os.path.basename(corpus_path).replace('.txt', '')}"
        )

        print(f"Running with: {args.model1} vs {args.model2} on {args.corpus_path}")
        main(args)
