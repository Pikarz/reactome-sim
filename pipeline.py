from __future__ import annotations

from pathlib import Path

import libsbml
import numpy as np
import roadrunner

import generate_sbml
import generate_target_file
import merge_sbml
import optimization


def run_merge(file1: str, file2: str, out_path: Path) -> tuple[Path, dict]:
    merged_doc, summary = merge_sbml.merge_sbml_with_summary(file1, file2)
    libsbml.SBMLWriter().writeSBMLToFile(merged_doc, str(out_path))
    model = merged_doc.getModel()
    return out_path, {
        "n_species": model.getNumSpecies(),
        "n_reactions": model.getNumReactions(),
        "summary": summary,
    }


def run_augment(sbml_path: Path, out_path: Path) -> tuple[Path, dict]:
    doc = generate_sbml._read_sbml_with_namespace_fix(sbml_path)
    model = doc.getModel()
    if model is None:
        raise ValueError("Invalid SBML: <model> missing.")
    stats = generate_sbml.augment_model(
        model, default_mean=0.5, epsilon=1e-6, threshold_m=1.0, k_default=0.1
    )
    generate_sbml.validate_document(doc)
    libsbml.SBMLWriter().writeSBMLToFile(doc, str(out_path))
    return out_path, stats


def _targets_cover_model(csv_path: Path, expected_species_ids: list[str]) -> bool:
    """True if the cached targets CSV contains every species the model expects.
    Guards against reusing a CSV generated for a different pathway pair."""
    import csv as _csv

    with open(csv_path, newline="", encoding="utf-8") as h:
        csv_species = {row["species_id"] for row in _csv.DictReader(h) if row.get("species_id")}
    return set(expected_species_ids).issubset(csv_species)


def run_generate_targets(
    sbml_path: Path, csv_path: Path, model: str = "llama3.1:8b", use_cache: bool = True,
    chunk_size: int = 30,
) -> tuple[Path, bool]:
    """Return (csv_path, cache_hit). Calls LLM only when cache is absent or disabled."""
    sbml_data = generate_target_file.parse_sbml(str(sbml_path))
    expected_species_ids = [s["species_id"] for s in sbml_data["species"]]

    # Reuse the cache only if it is the rich CSV (concentration/volume columns) AND
    # its species match this model. A stale CSV left over from a different pathway
    # pair would otherwise be reused (cache keyed by filename), feeding the optimizer
    # species that do not exist in the model -> every simulation fails -> the loss
    # collapses to the CVODE penalty plateau (~1e12).
    if (
        use_cache
        and csv_path.exists()
        and generate_target_file.csv_has_reference_columns(str(csv_path))
        and _targets_cover_model(csv_path, expected_species_ids)
    ):
        return csv_path, True
    # Chunked generation: ask the LLM for concentrations a few dozen species at a
    # time (and volumes once), so large models do not truncate one oversized JSON.
    concentration_map, volume_map = generate_target_file.generate_targets_chunked(
        sbml_data, model, temperature=0.2, chunk_size=chunk_size
    )
    rows = generate_target_file.build_target_rows(sbml_data["species"], concentration_map, volume_map)
    generate_target_file.write_csv(rows, str(csv_path))
    return csv_path, False


def run_inject_references(sbml_path: Path, targets_csv: Path) -> dict:
    """Write reference concentrations (Hill thresholds M, target means mu) and
    compartment volumes from the targets CSV into the augmented SBML. Must run
    after target generation and before optimization."""
    concentration_map, volume_map = generate_target_file.read_reference_maps(str(targets_csv))
    return generate_sbml.inject_references_into_file(
        str(sbml_path), concentration_map, volume_map
    )


def _smart_init(tunable_params: list[str], species_ids: list[str], targets: np.ndarray) -> np.ndarray:
    """Initialize each log_K_* param near a value that drives its species toward its target.

    Heuristic: at steady state, dS/dt = K_in − K_out * S → S = K_in / K_out.
    Assuming the partner parameter starts at 1, setting log_K_in_<sid> = log10(target_<sid>)
    or log_K_out_<sid> = −log10(target_<sid>) gets the SS near target before optimization
    starts. Falls back to 0 (i.e., K=1) for params that do not map to a known species.
    """
    target_map = dict(zip(species_ids, np.asarray(targets, dtype=float)))
    init = np.zeros(len(tunable_params), dtype=float)
    for i, pid in enumerate(tunable_params):
        if pid.startswith("log_k_rxn_"):
            # Inizializziamo k a 0.01 (10^-2) per evitare esplosioni a catena
            init[i] = -2.0

        for prefix, sign in (("log_K_in_", 1.0), ("log_K_out_", -1.0)):
            if pid.startswith(prefix):
                sid = pid[len(prefix):]
                t = target_map.get(sid)
                if t is not None and t > 1e-6:
                    # Clip to [-6, 6] so a tiny target does not push K below 1e-6
                    # (which produces a stiff ODE that CVODE struggles to integrate).
                    init[i] = float(np.clip(sign * np.log10(t), -6.0, 6.0))
                break
    return init


def run_optimize(
    sbml_path: Path,
    targets_csv: Path,
    tunable_params: list[str],
    sim_end: float,
    iterations: int,
    population_size: int = 20,
    learning_rate: float = 0.01,
    seed: int = 7,
    target_loss: float | None = None,
    n_workers: int = 1,
) -> tuple[np.ndarray, list[float], list[str], np.ndarray]:
    species_ids, targets = optimization.load_targets(str(targets_csv))
    print(f"Optimizing {len(tunable_params)} parameters to fit {len(species_ids)} targets...")
    init_log_params = _smart_init(tunable_params, species_ids, targets)
    optimization._SBML_PATH = str(sbml_path)
    best_params, loss_history = optimization.openai_es_minimize(
        init_log_params=init_log_params,
        parameter_ids=tunable_params,
        species_ids=species_ids,
        targets=targets,
        learning_rate=learning_rate,
        sigma=0.5,
        sim_start=0.0,
        sim_end=sim_end,
        iterations=iterations,
        population_size=population_size,
        seed=seed,
        target_loss=target_loss,
        n_workers=n_workers,
    )
    log_results = dict(zip(tunable_params, np.log10(best_params)))
    optimization.write_optimized_params_to_sbml(str(sbml_path), log_results)
    return best_params, loss_history, species_ids, targets


def run_simulate(
    sbml_path: Path, species_ids: list[str], end_time: float, points: int = 1000
) -> np.ndarray:
    rr = roadrunner.RoadRunner(str(sbml_path))
    rr.timeCourseSelections = ["time", *species_ids]
    return rr.simulate(0, end_time, points)
