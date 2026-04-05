import os
from typing import List, Tuple
import numpy as np
import roadrunner
import matplotlib.pyplot as plt


_HERE = os.path.dirname(os.path.abspath(__file__))
_SBML_PATH = os.path.join(_HERE, "working_homo-sapiens", "R-HSA-1855192.sbml")


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
    params = 10**log_params # Map base-10 log parameters back to raw values
    for pid, value in zip(parameter_ids, params):  # Assign each parameter value to the RoadRunner model.
        rr[pid] = float(value) 

    rr.reset()
    yi = simulate_terminal_means(rr, species_ids, sim_start, sim_end)  # Simulate to get y_i.
    return float(np.sum((yi - targets) ** 2))


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
    learning_rate: float = 0.05,
    seed: int = 7,
) -> tuple[np.ndarray, list[float]]:
    # Perform OpenAI-ES style optimization (mirrored/antithetic sampling) on log-parameters.
    # High-level idea:
    # 1) sample noise vectors eps_k
    # 2) evaluate objective at theta +/- sigma * eps_k
    # 3) convert objective (minimize) into scores (maximize) and estimate gradient in log-space
    # 4) take a gradient-ascent step on the score (equivalently descent on the objective)
    if population_size < 2:  # Need at least one mirrored pair; otherwise the estimator is ill-defined.
        raise ValueError("population_size must be >= 2") 

    rng = np.random.default_rng(seed)  # Create a reproducible random generator for sampling noise.
    theta = init_log_params.copy()  # Initialize theta (log-parameters) from the provided starting point.
    history = [] 
    half = population_size // 2  # We sample half and mirror to get population_size evaluations.

    rr = roadrunner.RoadRunner(_SBML_PATH)  # Build a fresh simulator instance from the SBML file.

    best_theta = theta.copy()  # Track best-seen theta (in log-space) over the whole run.
    best_f = objective_function(  # Evaluate the objective at the initial candidate.
        rr,
        best_theta, 
        parameter_ids, 
        species_ids,
        targets, 
        sim_start,
        sim_end, 
    ) 

    for step in range(iterations):
        eps = rng.standard_normal((half, theta.size))  # Sample Gaussian noise vectors.

        all_noise = []  # Collect noise vectors used in evaluations.
        all_scores = []  # Collect corresponding scores for ES weighting.

        for e in eps:  # Iterate over sampled noise directions.
            theta_plus = theta + sigma * e  # Positive perturbation in log-parameter space.
            theta_minus = theta - sigma * e  # Negative (mirrored) perturbation in log-parameter space.

            f_plus = objective_function( 
                rr,
                theta_plus,  
                parameter_ids,  
                species_ids,  
                targets,  
                sim_start,  
                sim_end
            )  
            f_minus = objective_function(  
                rr,
                theta_minus,  
                parameter_ids,  
                species_ids, 
                targets,  
                sim_start,  
                sim_end
            )  

            score_plus = -f_plus  # Convert minimization into maximization score for ES update.
            score_minus = -f_minus  # Same conversion for the mirrored evaluation.

            all_noise.append(e) 
            all_scores.append(score_plus)  
            all_noise.append(-e)  
            all_scores.append(score_minus) 

        noise_mat = np.vstack(all_noise)  # Stack noise vectors into shape (population_size, dim).
        scores = np.array(all_scores, dtype=float)  # Convert list of scores into a float numpy array.

        scores = (scores - scores.mean()) / (scores.std() + 1e-8)  # Normalize scores for stable learning.

        grad_estimate = (scores[:, None] * noise_mat).mean(axis=0) / sigma  # ES gradient estimate.
        theta = theta + learning_rate * grad_estimate  # Update theta in direction that increases score.

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

        if current_f < best_f:  # Check whether we found an improved (lower) objective value.
            best_f = current_f 
            best_theta = theta.copy()

        if step % 10 == 0 or step == iterations - 1:  # Periodically print progress (and always at the end).
            print(f"iter={step:03d}  F={current_f:.6f}  bestF={best_f:.6f}")  # Status line.

    return 10**best_theta, history

if __name__ == '__main__':
    rr_init = roadrunner.RoadRunner(_SBML_PATH) # initialize road runner

    observable_species = rr_init.model.getFloatingSpeciesIds() # get the observable species
    print(f"Species to optimize: {observable_species}") 

    params_to_tune = ["K_in", "K_out", "lambda_1"] # parameters to tune
    print(f"Parameters to tune: {params_to_tune}")

    species_ids, target_values = load_targets('./test.csv') # load the targets

    init_log_params = np.random.uniform(-6, 6, size=len(params_to_tune))
    print(f"Initial raw values: {10**init_log_params}")

    print("\n--- Starting OpenAI-ES Optimization ---")

    best_params, loss_history = openai_es_minimize(
        init_log_params=init_log_params,
        parameter_ids=params_to_tune,
        species_ids=species_ids,
        targets=target_values,
        learning_rate=0.01,
        sim_start=0.0,
        sim_end=1000.0,  
        iterations=1000
    )

    print("\n--- Optimization Complete ---")
    print(f"Final Best Loss: {loss_history[-1]:.6f}")
    for name, value in zip(params_to_tune, best_params):
        print(f"Optimized {name}: {value:.4f}")

    # plot the Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Objective Function $F$")
    plt.yscale('log') 
    plt.xlabel("Iteration")
    plt.ylabel("Error (Mean Squared)")
    plt.title("Optimization Progress (OpenAI-ES)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()