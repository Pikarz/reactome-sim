import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completa: Merge -> Augment -> Target -> Optimize -> Simulate"
    )
    parser.add_argument("--file1", required=True, help="Primo file SBML (obbligatorio)")
    parser.add_argument("--file2", default=None, help="Secondo file SBML (opzionale)")
    parser.add_argument("--output-dir", default="pipeline_output", help="Cartella di output")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="Modello Ollama per i target")
    parser.add_argument("--sim-end", type=float, default=100000.0, help="Orizzonte di simulazione")
    parser.add_argument("--iterations", type=int, default=10000, help="Iterazioni OpenAI-ES")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- INIZIO PIPELINE ---")

    if args.file2:
        print(f"\n[Step 0] Merge {args.file1} + {args.file2}")
        merged_path, info = pipeline.run_merge(args.file1, args.file2, out_dir / "merged.sbml")
        print(f"  → {merged_path} ({info['n_species']} species, {info['n_reactions']} reactions)")
    else:
        print(f"\n[Step 0] No merge, using {args.file1}")
        merged_path = Path(args.file1)

    print("\n[Step 1] Augment")
    aug_path, stats = pipeline.run_augment(
        merged_path, out_dir / f"{merged_path.stem}_augmented.sbml"
    )
    print(
        f"  → {aug_path}  "
        f"kinetic_laws={stats['kinetic_laws_added']}, "
        f"sources={stats['source_reactions_added']}, "
        f"sinks={stats['sink_reactions_added']}"
    )

    print(f"\n[Step 2] Generate targets via LLM ({args.llm_model})")
    csv_path, cache_hit = pipeline.run_generate_targets(
        aug_path, out_dir / "targets.csv", model=args.llm_model
    )
    print(f"  → {csv_path} ({'cache hit' if cache_hit else 'fresh LLM call'})")

    print(f"\n[Step 3] Optimize ({len(stats['tunable_params'])} params, {args.iterations} iters)")
    best_params, loss_history, species_ids, target_values = pipeline.run_optimize(
        aug_path,
        csv_path,
        stats["tunable_params"],
        sim_end=args.sim_end,
        iterations=args.iterations,
    )
    print(f"  Final loss: {loss_history[-1]:.6f}")

    horizon = args.sim_end * 1.5
    print(f"\n[Step 4] Simulate to t={horizon}")
    result = pipeline.run_simulate(aug_path, species_ids, horizon, points=10000)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_ids)))
    for i, sp_id in enumerate(species_ids):
        plt.plot(result[:, 0], result[:, i + 1], label=sp_id, color=colors[i], linewidth=2)
        plt.axhline(
            y=target_values[i], color=colors[i], linestyle="--", alpha=0.6,
            label=f"Target {sp_id}",
        )
    plt.xlabel("Tempo")
    plt.ylabel("Concentrazione / Amount")
    plt.title("Conferma di Stabilizzazione Modello Ottimizzato")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "validation_plot.png"
    plt.savefig(plot_path)
    print(f"\nPlot → {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
