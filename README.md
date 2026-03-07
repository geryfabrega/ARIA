# ARIA — Automated Red-teaming & Iterative Attack

ARIA is a custom jailbreak red-teaming pipeline built on top of [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench). It runs a mutating attack loop against LLM target models and benchmarks multiple jailbreak judges against human ground truth.

## Repository structure

```
ARIA/
├── jailbreakbench/          # git submodule — upstream JailbreakBench library
├── pipeline/                # custom pipeline code
│   ├── attack_pipeline/     # core modules (attacker, judge, target, feedback)
│   ├── run_attack.py        # entry point: mutating attack loop
│   ├── run_judge_comparison.py  # entry point: judge accuracy benchmark
│   └── test_query.py        # quick smoke-test against gpt-4o-mini
├── outputs/                 # generated CSVs (gitignored, .gitkeep tracks the folder)
├── .env.example             # template — copy to .env and fill in keys
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/geryfabrega/ARIA.git
cd ARIA
```

If you already cloned without `--recurse-submodules`, initialise the submodule manually:

```bash
git submodule update --init --recursive
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the JailbreakBench library from the submodule first, then the pipeline requirements:

```bash
pip install -e jailbreakbench/
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   TOGETHERAI_API_KEY=...
#   OPENAI_API_KEY=...
```

## Running the pipeline

All scripts are run from the **project root**.

### Mutating attack loop

```bash
python pipeline/run_attack.py                          # 10 behaviors, 3 cycles each
python pipeline/run_attack.py --behaviors 5            # first 5 behaviors
python pipeline/run_attack.py --behaviors 5 --max-cycles 5
python pipeline/run_attack.py --output outputs/my_run.csv
```

Results are written to `outputs/attack_results.csv` by default.

### Judge comparison benchmark

```bash
python pipeline/run_judge_comparison.py                # 30 samples
python pipeline/run_judge_comparison.py --samples 50
python pipeline/run_judge_comparison.py --output outputs/my_judge_run.csv
```

Results (per-example + summary) are written to `outputs/judge_comparison_results*.csv` by default.

### Quick smoke test

```bash
python pipeline/test_query.py
```

## Notes

- `outputs/` is gitignored — results are local only. Commit specific result files manually if you want to share them.
- The `jailbreakbench/` submodule tracks the upstream repo at a pinned commit. To update it: `git submodule update --remote jailbreakbench`.
- Never commit `.env`.
