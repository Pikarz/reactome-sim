# reactome-sim

This repository builds a simple workflow for SBML pathway simulation and parameter fitting with `libRoadRunner`.

## Project Workflow

1. Start from an SBML model in `working_homo-sapiens/`.
2. Generate target values for all model species with `generate_target_file.py` via local Ollama.
3. Run optimization with `roadrunner_new.py` to fit selected model parameters against target values.
4. (Optional) Save optimized parameter values back into the SBML file.

## Repository Structure

- `working_examples/`: Generic SBML templates for testing and experimentation.
- `working_homo-sapiens/`: Pathway SBML models prepared for dynamic simulation.
- `generated_target/`: Auto-generated prompts and CSV targets (created by `generate_target_file.py`).
- `generate_target_file.py`: Parses SBML, builds a prompt, calls local Ollama, and writes a target CSV.
- `roadrunner_new.py`: Runs OpenAI-ES style optimization with RoadRunner against target values.
- `test.csv`: Example target file used by `roadrunner_new.py` in its current configuration.

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`
- Ollama installed locally
- A pulled Ollama model (default in script: `llama3.2:3b`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Install and Run Ollama (Windows)

### 1) Install Ollama

Preferred (winget):

```powershell
winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
```

If `winget` is unavailable, install from: https://ollama.com/download/windows

### 2) Open a new PowerShell and verify CLI

```powershell
ollama --version
```

If you get `ollama non riconosciuto`, use the full path:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
```

### 3) Start the Ollama server

```powershell
ollama serve
```

If `ollama` is not in PATH, use:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

Keep this terminal open while generating targets.

If you see `bind ... 11434`, it means the server is already running (this is okay).

### 4) Pull the model

In a second terminal:

```powershell
ollama pull llama3.2:3b
```

PATH fallback:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2:3b
```

### 5) Verify server and models

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

You should see your downloaded model in the JSON response.

## How To Run `generate_target_file.py`

### 1) Ensure Ollama server is running and model is pulled

Server:

```powershell
ollama serve
```

Model:

```powershell
ollama pull llama3.2:3b
```

### 2) Generate prompt + targets from SBML

```bash
python generate_target_file.py --sbml working_homo-sapiens/R-HSA-1855192.sbml
```

Default outputs:

- `generated_target/R-HSA-1855192/prompt_R-HSA-1855192.txt`
- `generated_target/R-HSA-1855192/target.csv`

### Useful options

- Choose a specific Ollama model:

```bash
python generate_target_file.py --sbml working_homo-sapiens/R-HSA-1855192.sbml --model llama3.2:3b
```

- Dry run (no API call, prompt only):

```bash
python generate_target_file.py --sbml working_homo-sapiens/R-HSA-1855192.sbml --dry-run
```

- Custom output CSV:

```bash
python generate_target_file.py --sbml working_homo-sapiens/R-HSA-1855192.sbml --output-csv test.csv
```

## How To Run `roadrunner_new.py`

`roadrunner_new.py` currently uses:

- SBML model: `working_homo-sapiens/R-HSA-1855192.sbml`
- Targets file: `./test.csv`
- Tuned parameters: `log_K_in`, `log_K_out`, `log_lambda_1`

Run:

```bash
python roadrunner_new.py
```

What it does:

1. Loads target values from `test.csv`.
2. Simulates the model with RoadRunner.
3. Optimizes parameters with an OpenAI-ES loop.
4. Plots optimization loss.
5. Writes optimized parameter values back into the SBML file.

## Practical End-to-End Example

1. Generate a target file:

```bash
python generate_target_file.py --sbml working_homo-sapiens/R-HSA-1855192.sbml --output-csv test.csv
```

2. Fit model parameters against that target file:

```bash
python roadrunner_new.py
```

## How To Run `merge_sbml.py`

Use `merge_sbml.py` to merge two SBML pathway files into a single deterministic output model.

### Basic usage

```bash
python merge_sbml.py working_homo-sapiens/R-HSA-1660537.sbml working_homo-sapiens/R-HSA-1660508.sbml -o working_homo-sapiens/merged_1660537_1660508.sbml
```

### What gets written

- Merged SBML file at the path passed to `-o`.
- Merge report JSON next to it with suffix `.merge_report.json`.

For the example above, outputs are:

- `working_homo-sapiens/merged_1660537_1660508.sbml`
- `working_homo-sapiens/merged_1660537_1660508_merge_report.json`

### Useful option

- Suppress validation warnings:

```bash
python merge_sbml.py working_homo-sapiens/R-HSA-1660537.sbml working_homo-sapiens/R-HSA-1660508.sbml -o working_homo-sapiens/merged_1660537_1660508.sbml --suppress-validation-warnings
```
