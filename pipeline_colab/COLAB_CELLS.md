# Google Colab Cells (Copy/Paste)

Use the cells below in order.

## Cell 1: Setup + dependencies
```bash
!pip -q install python-dotenv litellm tenacity datasets jailbreakbench pandas
```

## Cell 2: Get code into Colab
Option A (recommended): clone your repo.
```bash
!git clone <YOUR_REPO_URL> /content/ARIA
%cd /content/ARIA
```

Option B: if you uploaded a zip, unpack and `cd` into the project root where `pipeline_colab/` exists.

## Cell 3: API keys + local model download
```python
import os
from getpass import getpass
from huggingface_hub import snapshot_download

os.environ["HF_TOKEN"] = getpass("HF_TOKEN (for model download): ")
os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY: ")

ATTACKER_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
os.environ["ATTACKER_MODEL_ID"] = ATTACKER_MODEL_ID
os.environ["TARGET_MODEL_ID"] = TARGET_MODEL_ID

print(f"Downloading attacker model: {ATTACKER_MODEL_ID} ...")
snapshot_download(repo_id=ATTACKER_MODEL_ID, token=os.environ["HF_TOKEN"])
print(f"Downloading target model: {TARGET_MODEL_ID} ...")
snapshot_download(repo_id=TARGET_MODEL_ID, token=os.environ["HF_TOKEN"])
print("Model downloads complete.")
```

## Cell 4: Run attack pipeline (function-style)
```python
import os
import sys

# Ensure imports like `from attack_pipeline...` resolve from pipeline_colab/
sys.path.insert(0, "/content/ARIA/pipeline_colab")

from pipeline_colab.colab_workflows import run_attack_workflow
import attack_pipeline.config as attack_config

attack_config.ATTACKER_MODEL = f"hf_local:{os.environ['ATTACKER_MODEL_ID']}"
attack_config.TARGET_MODEL = f"hf_local:{os.environ['TARGET_MODEL_ID']}"

attack_csv = run_attack_workflow(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    behaviors=10,
    max_cycles=3,
    output="outputs/attack_results_colab.csv",
)

attack_csv
```

## Cell 5: Preview attack output
```python
import pandas as pd

pd.read_csv("outputs/attack_results_colab.csv").head(20)
```

## Cell 6: Run judge comparison (optional)
```python
import os

from pipeline_colab.colab_workflows import run_judge_comparison_workflow

details_csv, summary_csv = run_judge_comparison_workflow(
    together_api_key=os.environ["TOGETHERAI_API_KEY"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    samples=30,
    output="outputs/judge_comparison_results_colab.csv",
)

(details_csv, summary_csv)
```

## Cell 7: Download outputs
```python
from google.colab import files

files.download("outputs/attack_results_colab.csv")
# files.download("outputs/judge_comparison_results_colab.csv")
# files.download("outputs/judge_comparison_results_colab_summary.csv")
```

## Optional CLI-style cells
If you prefer script execution instead of function calls:
```bash
!python pipeline_colab/run_attack.py --behaviors 5 --max-cycles 5 --output outputs/my_run.csv
!python pipeline_colab/run_judge_comparison.py --samples 30 --output outputs/judge_comparison_results.csv
```
