import os
from typing import List, Tuple
import numpy as np
import roadrunner
import xml.etree.ElementTree as ET


_HERE = os.path.dirname(os.path.abspath(__file__))
# Default SBML path; the pipeline overrides this per run via optimization._SBML_PATH.
_SBML_PATH = os.path.join(_HERE, "working_homo-sapiens", "R-HSA-1855192_augmented.sbml")


# --- Parallel population evaluation -------------------------------------------
# Each ES iteration evaluates population_size independent candidate parameter sets.
# These are embarrassingly parallel: we farm them across worker processes, each
# holding its own RoadRunner instance. The objective value of a candidate does not
# depend on the others, so parallelism yields results identical to the serial path.

_W = {}  # per-worker state (one RoadRunner per process)


def _worker_init(sbml_path, parameter_ids, species_ids, targets, sim_start, sim_end):
    # Build one RoadRunner per worker process and stash the fixed evaluation context.
    _W["rr"] = roadrunner.RoadRunner(sbml_path)
    _W["parameter_ids"] = parameter_ids
    _W["species_ids"] = species_ids
    _W["targets"] = np.asarray(targets, dtype=float)
    _W["sim_start"] = sim_start
    _W["sim_end"] = sim_end


def _worker_eval(log_params):
    # Evaluate one candidate using this worker's RoadRunner (same math as serial).
    return objective_function(
        _W["rr"],
        log_params,
        _W["parameter_ids"],
        _W["species_ids"],
        _W["targets"],
        _W["sim_start"],
        _W["sim_end"],
    )


def load_targets(path: str) -> Tuple[List[str], np.ndarray]:
    # Load targets by reading the file line-by-line.
    # Each non-empty line should contain: species_id and target_value.
    # Supported separators inside a line: comma.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Targets file not found: {path}")

    species_ids: List[str] = []  # Accumulate species identifiers (first column) in input order.
    targets: List[float] = []  # Accumulate numeric target values (second column) aligned with species_ids.
    header_checked = False  # Track whether we already processed the potential header row.

    with open(path, "r", newline="") as handle:
        for raw_line in handle:  # Iterate over the file one line at a time.
            line = raw_line.strip()  # Remove surrounding whitespace and the trailing newline.
            if not line:  # Skip empty/blank lines.
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",")]

            if not header_checked:  # Only the first line can be a header.
                header_checked = True  # Mark that we've checked the header condition.
                if len(parts) >= 2 and parts[0].lower() in {"species", "species_id"}: 
                    continue  # Skip header.

            if len(parts) < 2:  # We need at least two fields to parse species + value.
                raise ValueError(f"Invalid row (expected 2 columns): {line}")

            species = parts[0].strip()  # First column: the species id.
            value_str = parts[1].strip()  # Second column: the numeric target as a string.
            try:  # Parse the target value as a float.
                value = float(value_str)
            except ValueError as exc:
                raise ValueError(f"Invalid target value for species '{species}': {value_str}") from exc

            species_ids.append(species)
            targets.append(value)

    if not species_ids:  # Ensure we actually collected at least one data row.
        raise ValueError("Targets file is empty or contains no valid rows.")

    return species_ids, np.array(targets, dtype=float)


def simulate_terminal_means(
    rr: roadrunner.RoadRunner,
    species_ids: list[str],
    start: float,
    end: float
) -> np.ndarray:
    # Run a time-course simulation and compute y_i as the mean over the final part of the trajectory.
    selections = ["time", *species_ids]  # Ask RoadRunner to return time plus the chosen species columns.
    rr.reset()  # Reset dynamic state/time to initial conditions while keeping current parameter values.

    points = 2 # we only need start and end points
    result = rr.simulate(start, end, points, selections=selections)  # Simulate and collect a result matrix.

    # result[-1] is the last row (the final time point)
    # [1:] skips the 'time' column to get only species values
    return result[-1, 1:]


def objective_function(
    rr,
    log_params: np.ndarray,
    parameter_ids: list[str],
    species_ids: list[str],
    targets: np.ndarray,
    sim_start: float,
    sim_end: float
) -> float:
    # Compute the least-squares objective: F(theta) = sum_i (y_i(theta) - M_i)^2.
    # Here theta is represented in log-space; we exponentiate to enforce positivity of model parameters.
    for pid, value in zip(parameter_ids, log_params):  # Assign each parameter value to the RoadRunner model.
        rr[pid] = float(value) 

    try:
        yi = simulate_terminal_means(rr, species_ids, sim_start, sim_end)
    except RuntimeError:
        # Se CVODE fallisce a causa della rigidezza estrema, applichiamo una penalità altissima (1e12).
        # Usiamo un numero finito e non float('inf') per non generare NaN quando ES calcola medie/std.
        return 1e12 + np.random.uniform(0, 1e5)
    # --- FINE MODIFICA ---

    # Log-space error: targets span many orders of magnitude (e.g. 9e-8 ... 1e-4
    # after the concentration*volume conversion), so we fit log10(sim) to
    # log10(target). This is scale-free (every species weighted equally) AND, unlike
    # a clipped relative error, never saturates: a species that is 5 orders of
    # magnitude off contributes err=5 with a live gradient, instead of a flat
    # clipped plateau that blinds the optimizer. The floor guards log10 against
    # zero/negative simulated values from solver noise.
    floor = 1e-30
    log_sim = np.log10(np.maximum(yi, floor))
    log_tgt = np.log10(np.maximum(targets, floor))
    errors = log_sim - log_tgt

    return float(np.sum(errors ** 2))

    
def openai_es_minimize(
    init_log_params: np.ndarray,
    parameter_ids: list[str],
    species_ids: list[str],
    targets: np.ndarray,
    sim_start: float,
    sim_end: float,
    iterations: int = 60,
    population_size: int = 20,
    sigma: float = 0.10,
    learning_rate: float = 0.1,
    seed: int = 7,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
    min_lr_frac: float = 0.3,  # lr floor as a fraction of learning_rate (cosine endpoint)
    sigma_decay: float = 0.999,
    min_sigma_frac: float = 0.2,
    target_loss: float | None = None,
    n_workers: int = 1,
) -> tuple[np.ndarray, list[float]]:

    if population_size < 2:
        raise ValueError("population_size must be >= 2")

    rng = np.random.default_rng(seed)
    theta = init_log_params.copy()
    history = []
    half = population_size // 2

    rr = roadrunner.RoadRunner(_SBML_PATH)

    # Optional process pool: each worker holds its own RoadRunner and evaluates a
    # share of the population in parallel. Results are identical to the serial path
    # (per-candidate objective is independent), only faster.
    pool = None
    if n_workers and n_workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(_SBML_PATH, parameter_ids, species_ids, targets, sim_start, sim_end),
        )
        print(f"Parallel ES: evaluating population across {n_workers} workers.")

    best_theta = theta.copy()
    best_f = objective_function(
        rr,
        best_theta,
        parameter_ids,
        species_ids,
        targets,
        sim_start,
        sim_end,
    )

    # Adam state
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    min_sigma = sigma * min_sigma_frac

    try:
      for step in range(1, iterations + 1):

        # Shrink the perturbation over time (floored) so the search explores widely
        # early and can fine-tune near the optimum instead of wandering away from it.
        sigma_t = max(sigma * (sigma_decay ** (step - 1)), min_sigma)

        eps = rng.standard_normal((half, theta.size))

        # Build the antithetic candidate list and the matching noise list, in order.
        all_noise = []
        candidates = []
        for e in eps:
            all_noise.append(e)
            candidates.append(theta + sigma_t * e)
            all_noise.append(-e)
            candidates.append(theta - sigma_t * e)

        # Evaluate the whole population, in parallel when a pool is available.
        if pool is not None:
            fvals = pool.map(_worker_eval, candidates)
        else:
            fvals = [
                objective_function(rr, c, parameter_ids, species_ids, targets, sim_start, sim_end)
                for c in candidates
            ]
        all_scores = [-f for f in fvals]

        noise_mat = np.vstack(all_noise)
        scores = np.array(all_scores, dtype=float)

        # rank-normalization tends to outperform z-score for ES
        ranks = scores.argsort().argsort()
        scores = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

        grad = (scores[:, None] * noise_mat).mean(axis=0) / sigma_t

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)

        # Cosine schedule spanning the whole iteration budget: lr glides smoothly
        # from `learning_rate` (first step) down to `min_lr` (final step), regardless
        # of how many iterations are run. This avoids the geometric decay's problem of
        # collapsing to the floor within ~a hundred steps and then staying flat.
        min_lr = learning_rate * min_lr_frac
        if iterations > 1:
            cos = 0.5 * (1.0 + np.cos(np.pi * (step - 1) / (iterations - 1)))
        else:
            cos = 1.0
        lr_t = min_lr + (learning_rate - min_lr) * cos

        theta += lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)

        current_f = objective_function(
            rr,
            theta,
            parameter_ids,
            species_ids,
            targets,
            sim_start,
            sim_end
        )

        history.append(current_f)

        if current_f < best_f:
            best_f = current_f
            best_theta = theta.copy()

        if step % 10 == 0 or step == iterations:
            print(
                f"iter={step:03d} "
                f"F={current_f:.6f} "
                f"bestF={best_f:.6f} "
                f"lr={lr_t:.5f}"
            )

        # Stop early once the best solution is good enough; no point spending the
        # remaining budget polishing an already-converged fit.
        if target_loss is not None and best_f <= target_loss:
            print(f"early stop at iter={step}: bestF={best_f:.6g} <= target_loss={target_loss:.6g}")
            break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return 10 ** best_theta, history


def write_optimized_params_to_sbml(sbml_path, param_map):
    """
    Updates parameter values in an SBML file using ElementTree.
    """
    # Register namespaces to prevent 'ns0:' prefixes in the output
    # SBML Level 3 Core namespace is the default
    prefix_map = {
        "": "http://www.sbml.org/sbml/level3/version1/core",
        "xhtml": "http://www.w3.org/1999/xhtml",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
        "vCard": "http://www.w3.org/2001/vcard-rdf/3.0#",
        "dcterms": "http://purl.org/dc/terms/",
        "bqbiol": "http://biomodels.net/biology-qualifiers/"
    }
    
    for prefix, uri in prefix_map.items():
        ET.register_namespace(prefix, uri)

    tree = ET.parse(sbml_path)
    root = tree.getroot()

    # The parameter tag is usually inside <model><listOfParameters>
    # We use a wildcard search for 'parameter' tags to be robust
    updated_count = 0
    for parameter in root.iter('{http://www.sbml.org/sbml/level3/version1/core}parameter'):
        p_id = parameter.get('id')
        if p_id in param_map:
            # Update the value attribute specifically
            new_val = str(param_map[p_id])
            parameter.set('value', new_val)
            print(f"Updated: {p_id} -> {new_val}")
            updated_count += 1

    if updated_count == len(param_map):
        tree.write(sbml_path, encoding='utf-8', xml_declaration=True)
        print(f"Successfully updated {updated_count} parameters to {sbml_path}")
    else:
        ns = "{http://www.sbml.org/sbml/level3/version1/core}"
        found_ids = [p.get('id') for p in root.iter(f"{ns}parameter")]
        missing = [k for k in param_map.keys() if k not in found_ids]
        print(f"Error: Only found {updated_count}/{len(param_map)} parameters.")
        print(f"Missing IDs in SBML: {missing}")
        print("File was NOT saved.")