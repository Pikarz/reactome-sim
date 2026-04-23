import requests
import json
import subprocess

def download_model(model_name):
    """Download di un modello OLLAMA"""
    print(f"Downloading {model_name}...")
    result = subprocess.run(['ollama', 'pull', model_name], 
                          capture_output=True, text=True)
    print("Download completed!")
    if result.stderr:
        print("Errori:", result.stderr)

recommended_models = [
    'gemma2:2b',      # 1.6GB - Veloce per testing
    'llama3.2:3b',    # 2.0GB - Bilanciato
    'phi3:3.8b',      # 2.3GB - Microsoft Phi-3
    'gemma2:9b',      # 5.4GB - Più potente
    'llama3.1:8b',    # 4.7GB - Meta Llama 3.1
    'mistral:7b',     # 4.1GB - Mistral AI
    'deepseek-r1:8b', # 
    'qwen3:14b',      # 9.3GB - Qwen 3
    'mistral-small:22b',      # 
]

#Scarica un modello adatto alla memoria disponibile
models_to_download = []
MODEL_TO_USE = 'llama3.2:3b'
download_model(MODEL_TO_USE)