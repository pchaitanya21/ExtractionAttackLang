## Setup Instructions 

The multilingual data has been stored in the Data_Extraction_data folder: and there are 4 scripts to split the data into batches, each script runs the multilingual prompts for the Pythia and GPT-Neo Model 

1. Clone the repo:
   ```bash
   git clone https://github.com/pchaitanya21/ExtractionAttackLang.git
   cd ExtractionAttackLang
2. Install Hatch if not installed
   ```bash
   pip install hatch
3. Download the data folder Data_Extraction_data from GoogleDrive: https://drive.google.com/drive/folders/1aYdJxkKCMiJwQGaIjQaFnjPRB0Mq32a8?usp=drive_link and place it in the local repo 
   ```bash
   ExtractionAttackLang/
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

6. If this doesnt work : 
   ```bash
   hatch shell



7. Then execute each script 
   ```bash
   python data_batch1.py/data_batch2.py....
