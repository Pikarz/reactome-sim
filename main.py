import argparse
import os
from pathlib import Path
import libsbml
import numpy as np
import matplotlib.pyplot as plt
import roadrunner

# Importa i moduli forniti (assicurati che siano nella stessa directory)
import merge_sbml
import generate_sbml
import generate_target_file
import optimization

def main():
    parser = argparse.ArgumentParser(description="Pipeline completa: Merge -> Augment -> Target -> Optimize -> Simulate")
    parser.add_argument("--file1", required=True, help="Primo file SBML (obbligatorio)")
    parser.add_argument("--file2", default=None, help="Secondo file SBML (opzionale, per il merge)")
    parser.add_argument("--output-dir", default="pipeline_output", help="Cartella di output per i file generati")
    parser.add_argument("--llm-model", default="llama3.2:3b", help="Modello Ollama da usare per i target (default: llama3.2:3b)")
    parser.add_argument("--sim-end", type=float, default=100000.0, help="Orizzonte di simulazione per l'ottimizzazione")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- INIZIO PIPELINE ---")
    
    # ==========================================
    # STEP 0: Merge (Opzionale)
    # ==========================================
    if args.file2:
        print(f"\n[Step 0] Merge in corso tra {args.file1} e {args.file2}...")
        merged_sbml_path = out_dir / "merged.sbml"
        merged_doc, summary = merge_sbml.merge_sbml_with_summary(args.file1, args.file2)
        
        # Scrivi il file unito
        writer = libsbml.SBMLWriter()
        writer.writeSBMLToFile(merged_doc, str(merged_sbml_path))
        current_sbml_path = merged_sbml_path
        print(f"Merge completato. File salvato in: {current_sbml_path}")
    else:
        print(f"\n[Step 0] --file2 non presente (no merge), procedo con: {args.file1}")
        current_sbml_path = Path(args.file1)

    # ==========================================
    # STEP 1: Augmentation
    # ==========================================
    print(f"\n[Step 1] Aumento del modello SBML...")
    augmented_sbml_path = out_dir / f"{current_sbml_path.stem}_augmented.sbml"
    
    # Usa le funzioni di generate_sbml
    doc = generate_sbml._read_sbml_with_namespace_fix(current_sbml_path)
    model = doc.getModel()
    if model is None:
        raise ValueError("SBML non valido: <model> mancante.")
        
    stats = generate_sbml.augment_model(
        model,
        default_mean=0.5,
        epsilon=1e-6,
        threshold_m=1.0,
        k_default=0.1
    )
    generate_sbml.validate_document(doc)
    
    writer = libsbml.SBMLWriter()
    writer.writeSBMLToFile(doc, str(augmented_sbml_path))
    print(f"Augmentation completata. Aggiunte {stats['kinetic_laws_added']} leggi cinetiche, {stats['source_reactions_added']} source, {stats['sink_reactions_added']} sink. File salvato in: {augmented_sbml_path}")

    # ==========================================
    # STEP 2: Generazione Target via LLM
    # ==========================================
    print(f"\n[Step 2] Estrazione target via LLM ({args.llm_model})...")
    targets_csv_path = out_dir / "targets.csv"
    
    sbml_data = generate_target_file.parse_sbml(str(augmented_sbml_path))
    prompt = generate_target_file.build_prompt(sbml_data)
    
    try:
        response_text = generate_target_file.call_ollama(prompt, args.llm_model, temperature=0.2)
        response_json = generate_target_file.extract_json_from_text(response_text)
        expected_species_ids = [s["species_id"] for s in sbml_data["species"]]
        rows = generate_target_file.validate_targets(response_json, expected_species_ids)
        
        generate_target_file.write_csv(rows, str(targets_csv_path))
        print(f"Target generati e salvati in: {targets_csv_path}")
    except Exception as e:
        print(f"ERRORE durante la chiamata all'LLM: {e}")
        print("Assicurati che Ollama sia in esecuzione.")
        return

    # ==========================================
    # STEP 3: Ottimizzazione dei parametri
    # ==========================================
    print("\n[Step 3] Ottimizzazione con OpenAI-ES...")
    species_ids, target_values = optimization.load_targets(str(targets_csv_path))
    params_to_tune = ["log_K_in", "log_K_out", "log_lambda_1"]
    init_log_params = np.random.uniform(-6, 6, size=len(params_to_tune))
    
    # Salva momentaneamente il path in optimization perché la funzione ES crea un RoadRunner locale
    # affinché optimization.py peschi il file giusto
    optimization._SBML_PATH = str(augmented_sbml_path) 
    
    best_params, loss_history = optimization.openai_es_minimize(
        init_log_params=init_log_params,
        parameter_ids=params_to_tune,
        species_ids=species_ids,
        targets=target_values,
        learning_rate=0.01,
        sim_start=0.0,
        sim_end=args.sim_end,
        iterations=2000,  
        population_size=20
    )
    
    print(f"Ottimizzazione completata. Loss finale: {loss_history[-1]:.6f}")
    
    log_results = dict(zip(params_to_tune, np.log10(best_params)))
    optimization.write_optimized_params_to_sbml(str(augmented_sbml_path), log_results)
    
    # ==========================================
    # STEP 4: Simulazione finale a orizzonte lungo
    # ==========================================
    extended_horizon = args.sim_end * 1.5
    print(f"\n[Step 4] Simulazione validazione fino a t={extended_horizon}...")
    
    rr_final = roadrunner.RoadRunner(str(augmented_sbml_path))
    selections = ["time", *species_ids]
    rr_final.timeCourseSelections = selections
    
    result = rr_final.simulate(0, extended_horizon, 10000)
    
    # Grafico delle concentrazioni vs target
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_ids)))
    
    for i, sp_id in enumerate(species_ids):
        # Traccia l'andamento reale
        plt.plot(result[:, 0], result[:, i+1], label=f"{sp_id}", color=colors[i], linewidth=2)
        # Traccia il target desiderato come linea tratteggiata
        plt.axhline(y=target_values[i], color=colors[i], linestyle='--', alpha=0.6, 
                    label=f"Target {sp_id}")

    plt.xlabel("Tempo")
    plt.ylabel("Concentrazione / Amount")
    plt.title("Conferma di Stabilizzazione Modello Ottimizzato")
    
    # Metti la legenda fuori per non coprire il grafico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = out_dir / "validation_plot.png"
    plt.savefig(plot_path)
    print(f"Grafico salvato in: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()