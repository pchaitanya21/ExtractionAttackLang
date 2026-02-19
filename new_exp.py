from run_batch import run_batch
from helper import get_data_folder
import os

folder = get_data_folder()
files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
paths = [os.path.join(folder, f) for f in files]
print(f"Queueing experiments for: {files}")
run_batch(paths)
