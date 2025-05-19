## Setup Instructions 

The multilingual data has been stored in the Data_exxtraction_data folder: and there are 4 scripts to split the data into batches, each script runs the multilingual prompts for the Pythia and GPT-Neo Model 

1. Clone the repo:
   ```bash
   git clone https://github.com/pchaitanya21/ExtractionAttackLang.git
   cd ExtractionAttackLang
2. Install Hatch if not installed
   ```bash
   pip install hatch
3. Download the data folder Data_Extraction_data from GoogleDrive:
   ```bash
   Data_Extraction_data/
   ├── Data_Extraction_data/     ← from Google Drive
   ├── data_run1.py
   ├── data_run2.py
   ├── helper.py
   ├── run_batch.py
   ├── pyproject.toml
4. Create the Hatch Environment
   ```bash
   hatch env create
5. Run the Scripts via Hatch
   ```bash
   hatch run run1
   hatch run run2
   hatch run run3
   hatch run run4
