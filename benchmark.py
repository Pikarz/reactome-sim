"""Scaling benchmark for the SBML pipeline: merge → augment → targets → optimize → simulate.

Measures wall time, CPU time, peak RSS, and per-stage correctness across SBML
inputs of increasing size. Caches LLM targets per scenario. Writes an output
directory with reports/ (CSV, Markdown summary, JSON dump) and
plots/{performance,optimization,simulation}/. Re-run with --replot to regenerate
reports and figures from a previous run's saved data without re-optimizing.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import libsbml
import matplotlib.pyplot as plt
import numpy as np
import roadrunner

import pipeline


# Hardcoded scaling scenarios. Each pair was checked for ≥1 shared species so the
# merge step is non-trivial. Ordered by combined input size (ascending).
SCENARIOS = [
    # {
    #    "name": "small",
    #    "file1": "homo_sapiens.3.1.sbml/R-HSA-1660508.sbml",
    #    "file2": "homo_sapiens.3.1.sbml/R-HSA-1660537.sbml",
    # },
    {
         "name": "medium",
         "file1": "homo_sapiens.3.1.sbml/R-HSA-1059683.sbml",
         "file2": "homo_sapiens.3.1.sbml/R-HSA-109703.sbml",
    },
  #  large: 156 merged species, 214 tunable params
    # {
    #     "name": "large",
    #     "file1": "homo_sapiens.3.1.sbml/R-HSA-111997.sbml",
    #     "file2": "homo_sapiens.3.1.sbml/R-HSA-9855142.sbml",
    # },
]

STAGE_ORDER = ["merge", "augment", "targets", "inject", "optimize", "simulate"]


# Metrics dataclasses ---------------------------------------------------------

@dataclass
class StageMetric:
    name: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0
    ok: bool = False
    detail: str = ""


@dataclass
class ScenarioResult:
    name: str
    file1: str
    file2: str | None
    input_size_bytes: int = 0
    merged_n_species: int = 0
    merged_n_reactions: int = 0
    merged_size_bytes: int = 0
    augmented_size_bytes: int = 0
    n_tunable_params: int = 0
    n_workers: int = 1
    initial_loss: float = float("nan")
    final_loss: float = float("nan")
    stages: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)
    targets_synthetic: bool = False


# Resource sampling -----------------------------------------------------------

def _proc_rss_bytes(pid: int) -> int:
    """RSS of a single process from /proc (Linux). 0 if unavailable."""
    try:
        with open(f"/proc/{pid}/statm") as h:
            pages = int(h.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _child_pids(pid: int) -> list[int]:
    """Direct child PIDs of a process via /proc/<pid>/task/<tid>/children."""
    kids: list[int] = []
    try:
        task_dir = f"/proc/{pid}/task"
        for tid in os.listdir(task_dir):
            try:
                with open(f"{task_dir}/{tid}/children") as h:
                    kids.extend(int(x) for x in h.read().split())
            except Exception:
                pass
    except Exception:
        pass
    return kids


def _read_rss_bytes() -> int:
    """Total RSS over the whole process tree (self + descendants), so the peak-memory
    measurement remains correct when the optimize stage spawns parallel workers.
    Returns 0 on platforms without procfs."""
    total = 0
    seen: set[int] = set()
    stack = [os.getpid()]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _proc_rss_bytes(pid)
        stack.extend(_child_pids(pid))
    return total

class _RssSampler(threading.Thread):
    def __init__(self, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self.interval = interval_s
        self.peak = _read_rss_bytes()
        self._stop_event = threading.Event() 

    def run(self):
        while not self._stop_event.is_set():
            rss = _read_rss_bytes()
            if rss > self.peak:
                self.peak = rss
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()


@contextmanager
def stage(name: str, collector: list):
    """Time a block and capture peak RSS. Swallows exceptions, recording them as ok=False."""
    gc.collect()
    sampler = _RssSampler()
    sampler.start()
    box = {"ok": True, "detail": ""}
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield box
    except Exception as exc:
        box["ok"] = False
        box["detail"] = f"exception: {type(exc).__name__}: {exc}"
    finally:
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        sampler.stop()
        collector.append(
            StageMetric(
                name=name,
                wall_s=wall,
                cpu_s=cpu,
                peak_rss_mb=sampler.peak / (1024 ** 2),
                ok=box["ok"],
                detail=box["detail"],
            )
        )


# Correctness checks ----------------------------------------------------------

def _species_ids(model: libsbml.Model) -> set[str]:
    return {model.getSpecies(i).getId() for i in range(model.getNumSpecies())}


def _floating_species_ids(model: libsbml.Model) -> list[str]:
    return [
        model.getSpecies(i).getId()
        for i in range(model.getNumSpecies())
        if not model.getSpecies(i).getBoundaryCondition()
        and not model.getSpecies(i).getConstant()
    ]


def _consistency_errors(doc: libsbml.SBMLDocument) -> int:
    doc.checkConsistency()
    return sum(
        1
        for i in range(doc.getNumErrors())
        if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    )


def check_merge(merged_path: Path, file1: str, file2: str) -> tuple[bool, str]:
    doc_m = libsbml.readSBML(str(merged_path))
    m_m = doc_m.getModel()
    if m_m is None:
        return False, "merged model is None"
    s_m = _species_ids(m_m)
    s1 = _species_ids(libsbml.readSBML(file1).getModel())
    s2 = _species_ids(libsbml.readSBML(file2).getModel())
    shared = s1 & s2
    if len(s_m) < len(s1 | s2):
        return False, f"merged species {len(s_m)} < union {len(s1 | s2)}"
    if not shared:
        return False, "no shared species (merge trivial)"
    if (errs := _consistency_errors(doc_m)):
        return False, f"{errs} SBML consistency errors"
    return True, f"shared={len(shared)} union={len(s1 | s2)} merged_species={len(s_m)}"


def check_augment(aug_path: Path, stats: dict) -> tuple[bool, str]:
    doc = libsbml.readSBML(str(aug_path))
    model = doc.getModel()
    if model is None:
        return False, "augmented model is None"
    for r in model.getListOfReactions():
        if not r.isSetKineticLaw():
            return False, f"reaction {r.getId()} missing kinetic law"
    produced, consumed = set(), set()
    for r in model.getListOfReactions():
        for sr in r.getListOfProducts():
            produced.add(sr.getSpecies())
        for sr in r.getListOfReactants():
            consumed.add(sr.getSpecies())
    for sid in _floating_species_ids(model):
        if sid not in produced:
            return False, f"{sid} no producer/source"
        if sid not in consumed:
            return False, f"{sid} no consumer/sink"
    if not stats.get("tunable_params"):
        return False, "tunable_params is empty"
    try:
        roadrunner.RoadRunner(str(aug_path))
    except Exception as exc:
        return False, f"RoadRunner load failed: {exc}"
    return True, f"tunable={len(stats['tunable_params'])}"


def check_targets(csv_path: Path, expected_ids: list[str]) -> tuple[bool, str]:
    if not csv_path.exists():
        return False, "CSV not found"
    seen: dict[str, float] = {}
    with open(csv_path, newline="") as h:
        for row in csv.DictReader(h):
            seen[row["species_id"]] = float(row["target_value"])
    missing = [i for i in expected_ids if i not in seen]
    if missing:
        return False, f"missing {len(missing)} targets (first: {missing[:3]})"
    if any(not np.isfinite(v) or v < 0 for v in seen.values()):
        return False, "non-finite or negative target value"
    return True, f"targets={len(seen)}"


def check_optimization(loss_history: list[float]) -> tuple[bool, str]:
    if not loss_history:
        return False, "empty loss history"
    if any(not np.isfinite(x) for x in loss_history):
        return False, "non-finite loss in history"
    delta = loss_history[0] - loss_history[-1]
    if delta <= 0:
        return False, f"loss did not decrease (Δ={delta:.4g})"
    rel = delta / max(abs(loss_history[0]), 1e-12)
    return True, f"final={loss_history[-1]:.4g} Δ={delta:.4g} ({rel * 100:.1f}%)"


def check_simulation(
    result: np.ndarray, species_ids: list[str], targets: np.ndarray, tol_rel: float
) -> tuple[bool, str, dict]:
    final = np.asarray(result)[-1, 1:]
    # Floor must stay far below real targets: after the concentration*volume conversion
    # abundances can be ~1e-7 or smaller, so a 1e-8 floor would understate the error of
    # tiny targets and let a big miss pass the tolerance. Match the optimizer's floor.
    rel_errors = {
        sid: float(abs(final[i] - targets[i]) / max(abs(targets[i]), 1e-12))
        for i, sid in enumerate(species_ids)
    }
    if not np.all(np.isfinite(final)):
        return False, "NaN/Inf in final state", rel_errors
    failed = [sid for sid, e in rel_errors.items() if e > tol_rel]
    if failed:
        return (
            False,
            f"{len(failed)}/{len(species_ids)} species exceed rel_err={tol_rel}",
            rel_errors,
        )
    return True, f"max_rel_err={max(rel_errors.values()):.3f}", rel_errors


# Synthetic target fallback ---------------------------------------------------

def _write_synthetic_targets(aug_path: Path, csv_path: Path, value: float = 0.5) -> list[str]:
    """Write a uniform-target CSV so optimize/simulate can still be benchmarked when LLM fails.

    Uses the rich column layout (concentration/compartment/volume) so the downstream
    reference-injection step can consume it just like a real LLM-generated CSV. With
    volume=1 the abundance target equals the concentration value.
    """
    model = libsbml.readSBML(str(aug_path)).getModel()
    sids = _floating_species_ids(model)
    with open(csv_path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["species_id", "target_value", "concentration", "compartment", "volume"])
        for sid in sids:
            comp = model.getSpecies(sid).getCompartment()
            w.writerow([sid, value, value, comp, 1.0])
    return sids


def _read_target_values(csv_path: Path) -> list[float]:
    with open(csv_path, newline="") as h:
        return [float(row["target_value"]) for row in csv.DictReader(h)]


# Scenario runner -------------------------------------------------------------

def run_scenario(sc: dict, args, out_root: Path) -> ScenarioResult:
    name = sc["name"]
    merged_path = out_root / f"{name}_merged.sbml"
    aug_path = out_root / f"{name}_augmented.sbml"
    targets_csv = out_root / f"{name}_targets.csv"

    file2 = sc.get("file2")
    res = ScenarioResult(name=name, file1=sc["file1"], file2=file2)
    res.n_workers = getattr(args, "workers", 1)
    res.input_size_bytes = Path(sc["file1"]).stat().st_size
    if file2:
        res.input_size_bytes += Path(file2).stat().st_size
    scenario_inputs = f"{sc['file1']} + {file2}" if file2 else f"{sc['file1']} (single input)"
    print(f"\n=== Scenario: {name} ({scenario_inputs}) ===")

    aug_stats: dict = {}
    species_ids: list[str] = []
    target_values = np.array([])

    if file2 and not args.skip_merge:
        with stage("merge", res.stages) as box:
            _, info = pipeline.run_merge(sc["file1"], file2, merged_path)
            res.merged_n_species = info["n_species"]
            res.merged_n_reactions = info["n_reactions"]
            res.merged_size_bytes = merged_path.stat().st_size
            box["ok"], box["detail"] = check_merge(merged_path, sc["file1"], file2)
        if not res.stages[-1].ok:
            print(f"  ✗ merge: {res.stages[-1].detail}")
            return res
        print(f"  ✓ merge: {res.stages[-1].detail} ({res.stages[-1].wall_s:.2f}s)")
        merge_input = merged_path
    else:
        res.stages.append(
            StageMetric(name="merge", ok=True, detail="skipped (single input)")
        )
        merge_input = Path(sc["file1"])

    with stage("augment", res.stages) as box:
        _, aug_stats = pipeline.run_augment(merge_input, aug_path)
        res.augmented_size_bytes = aug_path.stat().st_size
        res.n_tunable_params = len(aug_stats.get("tunable_params", []))
        box["ok"], box["detail"] = check_augment(aug_path, aug_stats)
    if not res.stages[-1].ok:
        print(f"  ✗ augment: {res.stages[-1].detail}")
        return res
    print(f"  ✓ augment: {res.stages[-1].detail} ({res.stages[-1].wall_s:.2f}s)")

    with stage("targets", res.stages) as box:
        try:
            pipeline.run_generate_targets(
                aug_path, targets_csv, model=args.llm_model, use_cache=True
            )
            expected = _floating_species_ids(libsbml.readSBML(str(aug_path)).getModel())
            ok, detail = check_targets(targets_csv, expected)
            if ok:
                vals = _read_target_values(targets_csv)
                # Degenerate output (all zero or all identical) is unusable: drive the
                # benchmark to the synthetic fallback so optimize/simulate still run.
                if max(vals) <= 0:
                    raise ValueError("LLM returned all-zero targets")
                if len(set(vals)) == 1:
                    raise ValueError(f"LLM returned identical targets ({vals[0]})")
            box["ok"], box["detail"] = ok, detail
        except Exception as exc:
            _write_synthetic_targets(aug_path, targets_csv, value=args.fallback_target)
            res.targets_synthetic = True
            box["ok"] = False
            box["detail"] = (
                f"LLM failed ({type(exc).__name__}: {exc}); "
                f"using synthetic targets={args.fallback_target}"
            )
    print(
        f"  {'✓' if res.stages[-1].ok else '⚠'} targets: {res.stages[-1].detail} "
        f"({res.stages[-1].wall_s:.2f}s)"
    )
    print(f"    (tunable_params={aug_stats.get('tunable_params', [])}, ")

    # Inject the reference concentrations (Hill thresholds M) and compartment volumes
    # from the targets CSV into the model, mirroring the main pipeline. Without this
    # the augmented model keeps placeholder M=1 and unset volumes, so the optimization
    # would run on a model that ignores points 1-3.
    with stage("inject", res.stages) as box:
        inj = pipeline.run_inject_references(aug_path, targets_csv)
        box["ok"] = True
        box["detail"] = (
            f"M set={inj['m_set']}, mu set={inj['mu_set']}, volumes set={inj['volumes_set']}"
        )
    print(
        f"  {'✓' if res.stages[-1].ok else '✗'} inject: {res.stages[-1].detail} "
        f"({res.stages[-1].wall_s:.2f}s)"
    )

    with stage("optimize", res.stages) as box:
        best_params, loss_history, species_ids, target_values = pipeline.run_optimize(
            aug_path,
            targets_csv,
            aug_stats["tunable_params"],
            sim_end=args.sim_end,
            iterations=args.iterations,
            population_size=args.population_size,
            learning_rate=args.learning_rate,
            target_loss=(args.early_stop_loss if args.early_stop_loss > 0 else None),
            n_workers=args.workers,
        )
        res.loss_history = [float(x) for x in loss_history]
        res.initial_loss = res.loss_history[0]
        res.final_loss = res.loss_history[-1]
        box["ok"], box["detail"] = check_optimization(loss_history)
    print(
        f"  {'✓' if res.stages[-1].ok else '✗'} optimize: {res.stages[-1].detail} "
        f"({res.stages[-1].wall_s:.2f}s)"
    )

    if not res.stages[-1].ok or not len(species_ids):
        # Optimize failed: record simulate as skipped rather than crashing on empty arrays.
        res.stages.append(
            StageMetric(
                name="simulate", ok=False,
                detail="skipped (depends on optimize)",
            )
        )
        print(f"  ⊘ simulate: skipped (optimize failed)")
        return res

    with stage("simulate", res.stages) as box:
        sim = pipeline.run_simulate(aug_path, species_ids, args.sim_end * 1.5, points=2000)
        ok, detail, _ = check_simulation(sim, species_ids, target_values, tol_rel=args.tol_rel)
        res.targets = {sid: float(target_values[i]) for i, sid in enumerate(species_ids)}
        res.final_state = {sid: float(sim[-1, i + 1]) for i, sid in enumerate(species_ids)}
        sim_arr = np.asarray(sim)
        np.savez_compressed(
            out_root / f"{name}_simulation.npz",
            time=sim_arr[:, 0],
            trajectories=sim_arr[:, 1:],
            species_ids=np.array(species_ids),
            targets=np.asarray(target_values, dtype=float),
        )
        # Stash for plotting without polluting dataclass fields (asdict ignores).
        res.sim_time = sim_arr[:, 0]
        res.sim_trajectories = sim_arr[:, 1:]
        res.sim_species_ids = list(species_ids)
        res.sim_targets = np.asarray(target_values, dtype=float)
        box["ok"], box["detail"] = ok, detail
    print(
        f"  {'✓' if res.stages[-1].ok else '✗'} simulate: {res.stages[-1].detail} "
        f"({res.stages[-1].wall_s:.2f}s)"
    )

    return res


# Reporting -------------------------------------------------------------------

def write_csv_report(results: list[ScenarioResult], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(
            ["scenario", "stage", "wall_s", "cpu_s", "peak_rss_mb", "ok", "detail"]
        )
        for r in results:
            for s in r.stages:
                w.writerow(
                    [r.name, s.name, f"{s.wall_s:.4f}", f"{s.cpu_s:.4f}",
                     f"{s.peak_rss_mb:.2f}", s.ok, s.detail]
                )


def write_markdown_summary(results: list[ScenarioResult], out_path: Path) -> None:
    lines = ["# Pipeline Scaling Benchmark\n"]
    lines.append("## Per-scenario summary\n")
    lines.append(
        "| Scenario | Inputs (MB) | Species | Reactions | Tunable | "
        "Wall total (s) | Peak RAM (MB) | Final loss | LLM targets | All ok |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for r in results:
        total = sum(s.wall_s for s in r.stages)
        peak = max((s.peak_rss_mb for s in r.stages), default=0.0)
        ok = all(s.ok for s in r.stages)
        llm_status = "FAIL (synthetic fallback)" if r.targets_synthetic else "OK"
        lines.append(
            f"| {r.name} | {r.input_size_bytes / 1024 / 1024:.2f} | {r.merged_n_species} "
            f"| {r.merged_n_reactions} | {r.n_tunable_params} | {total:.2f} "
            f"| {peak:.1f} | {r.final_loss:.4g} | {llm_status} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )

    lines.append("\n## Per-stage wall time (s)\n")
    lines.append("| Scenario | " + " | ".join(STAGE_ORDER) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(STAGE_ORDER)) + "|")
    for r in results:
        row = {s.name: s for s in r.stages}
        cells = [f"{row[k].wall_s:.2f}" if k in row else "—" for k in STAGE_ORDER]
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")

    lines.append("\n## Per-stage peak RSS (MB)\n")
    lines.append("| Scenario | " + " | ".join(STAGE_ORDER) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(STAGE_ORDER)) + "|")
    for r in results:
        row = {s.name: s for s in r.stages}
        cells = [f"{row[k].peak_rss_mb:.1f}" if k in row else "—" for k in STAGE_ORDER]
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")

    lines.append("\n## Correctness detail\n")
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("| Stage | ok | detail |")
        lines.append("|---|:---:|---|")
        for s in r.stages:
            lines.append(f"| {s.name} | {'PASS' if s.ok else 'FAIL'} | {s.detail} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# Plots -----------------------------------------------------------------------

_STAGE_COLORS = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(STAGE_ORDER)))


def _apply_plot_style() -> None:
    """Paper-quality, scientific defaults for every figure: serif text with
    Computer-Modern math, real LaTeX when the toolchain is available (falls back to
    matplotlib mathtext otherwise), restrained grid, no top/right spines."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })
    # We deliberately keep matplotlib's mathtext (Computer-Modern) rather than a real
    # LaTeX backend: it renders the same paper-style math without depending on a
    # latex/dvipng install and without needing to escape % or underscores that appear
    # in dynamic labels (e.g. species ids).


def _annotate_bars(ax, bars, values, fmt="{:.2f}"):
    for bar, v in zip(bars, values):
        if v <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(v),
            ha="center", va="bottom", fontsize=7,
        )


def _grouped_bar(results, attr: str, ylabel: str, title: str, out_path: Path,
                 logy: bool = True, fmt: str = "{:.2f}") -> None:
    scenarios = [r.name for r in results]
    x = np.arange(len(scenarios), dtype=float)
    width = 0.8 / len(STAGE_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, sname in enumerate(STAGE_ORDER):
        vals = [
            next((getattr(s, attr) for s in r.stages if s.name == sname), 0.0)
            for r in results
        ]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=sname,
            color=_STAGE_COLORS[i],
            edgecolor="black",
            linewidth=0.4,
        )
        _annotate_bars(ax, bars, vals, fmt=fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Stage", loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    if logy:
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loss_curves(results: list[ScenarioResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("plasma")
    for i, r in enumerate(results):
        if not r.loss_history:
            continue
        color = cmap(0.12 + 0.7 * i / max(1, len(results) - 1))
        hist = r.loss_history
        n = len(hist)
        # Plot the per-iteration loss faintly, plus the running-best as a bold line so
        # the actual convergence (what the optimizer keeps) reads clearly through noise.
        running_best = np.minimum.accumulate(hist)
        ax.plot(hist, color=color, linewidth=0.8, alpha=0.25)
        ax.plot(running_best, color=color, linewidth=2.2, alpha=0.95,
                label=f"{r.name}  (species={r.merged_n_species}, "
                      f"best={min(hist):.2g}, stopped@{n})")
        # Mark the best point reached.
        bi = int(np.argmin(hist))
        ax.scatter([bi], [hist[bi]], color=color, s=45, zorder=5,
                   edgecolor="black", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss — Σ (log₁₀ sim − log₁₀ target)²  (log scale)")
    ax.set_title("Optimization convergence (OpenAI-ES)\nfaint = per-iteration loss, bold = running best")
    ax.legend(loc="upper right", title="Scenario")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _log_errors(targets: dict, final_state: dict):
    """Aligned arrays (species, target, simulated, |log10 dex error|)."""
    sids = list(targets)
    tg = np.array([targets[s] for s in sids], dtype=float)
    fs = np.array([final_state.get(s, np.nan) for s in sids], dtype=float)
    le = np.abs(np.log10(np.maximum(fs, 1e-30)) - np.log10(np.maximum(tg, 1e-30)))
    return sids, tg, fs, le


def plot_final_errors(results: list[ScenarioResult], out_path: Path) -> None:
    """Fit quality across ALL species: (A) simulated-vs-target scatter on log-log
    axes with a +/-1 dex band, and (B) the ECDF of per-species log-error."""
    plottable = [r for r in results if r.targets and r.final_state]
    if not plottable:
        return
    n = len(plottable)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 4.3 * n), layout="constrained", squeeze=False)

    for row, r in enumerate(plottable):
        _, tg, fs, le = _log_errors(r.targets, r.final_state)
        m = np.isfinite(tg) & np.isfinite(fs)
        tgm, fsm, lem = tg[m], fs[m], le[m]
        suffix = " (synthetic targets)" if r.targets_synthetic else ""

        # ---- (A) simulated vs target, log-log ----
        axA = axes[row, 0]
        lo = max(min(tgm.min(), fsm[fsm > 0].min() if np.any(fsm > 0) else tgm.min()) * 0.3, 1e-30)
        hi = max(tgm.max(), fsm.max()) * 3
        diag = np.array([lo, hi])
        axA.fill_between(diag, diag / 10, diag * 10, color="0.85", alpha=0.7, lw=0,
                         label=r"$\pm 1$ dex")
        axA.plot(diag, diag, color="0.30", ls="--", lw=1.0, label=r"$y = x$")
        sc = axA.scatter(tgm, np.maximum(fsm, lo), c=lem, cmap="viridis", vmin=0, vmax=2,
                         s=20, edgecolor="black", linewidth=0.2, zorder=3)
        axA.set_xscale("log"); axA.set_yscale("log")
        axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
        axA.set_xlabel("Target concentration (a.u.)")
        axA.set_ylabel("Simulated steady state (a.u.)")
        axA.set_title(f"{r.name}{suffix}: simulated vs. target ({m.sum()} species)")
        axA.legend(loc="upper left")
        cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
        cb.set_label(r"$|\log_{10}(\mathrm{sim}/\mathrm{target})|$  (dex)")

        # ---- (B) ECDF of per-species log-error ----
        axB = axes[row, 1]
        e = np.sort(lem)
        y = np.arange(1, e.size + 1) / e.size
        axB.step(e, y, where="post", color="#08306b", lw=1.8)
        axB.fill_between(e, 0, y, step="post", color="#4c78a8", alpha=0.15)
        lines = []
        for thr, c in ((0.5, "#2ca02c"), (1.0, "#d62728")):
            frac = float((lem <= thr).mean())
            axB.axvline(thr, color=c, ls=":", lw=1.2)
            lines.append((f"$\\leq {thr:g}$ dex:  {frac * 100:.0f}%", c))
        # Compact stats box (bottom-right), one coloured line per threshold.
        for k, (txt, c) in enumerate(lines):
            axB.text(0.97, 0.10 + 0.08 * k, txt, transform=axB.transAxes,
                     color=c, fontsize=9, ha="right", va="bottom")
        axB.set_xlim(0, max(2.0, float(e.max())))
        axB.set_ylim(0, 1.02)
        axB.set_xlabel(r"Per-species error  $|\log_{10}(\mathrm{sim}/\mathrm{target})|$  (dex)")
        axB.set_ylabel("Cumulative fraction of species")
        axB.set_title(f"{r.name}{suffix}: error distribution (median {np.median(lem):.2f} dex)")

    fig.savefig(out_path)
    plt.close(fig)


def plot_simulation_trajectories(
    results: list[ScenarioResult], out_root: Path, max_species: int = 60
) -> None:
    """Per-scenario time-course of the optimized model: one line per species with its
    target marked. Only emitted for scenarios small enough to read (<= max_species);
    larger merges have too many overlapping trajectories to be legible and are
    skipped (their convergence is summarized by the optimization plots instead)."""
    for r in results:
        time = getattr(r, "sim_time", None)
        traj = getattr(r, "sim_trajectories", None)
        sids = getattr(r, "sim_species_ids", None)
        tgts = getattr(r, "sim_targets", None)
        if time is None or traj is None or not sids:
            continue
        n = len(sids)
        if n > max_species:
            print(f"  simulation plot skipped for {r.name}: {n} species > {max_species} (too dense)")
            continue

        time = np.asarray(time, dtype=float)
        traj = np.asarray(traj, dtype=float)
        tg = np.asarray(tgts, dtype=float)
        suffix = " (synthetic targets)" if r.targets_synthetic else ""

        # Plot each species relative to its own target. This removes the 8+ decade
        # spread of absolute values (so trajectories no longer "all look the same"
        # rising from ~0) and replaces a column of overlapping target markers with a
        # single reference line at 1: every well-fit species settles onto it.
        ratio = np.maximum(traj, 1e-30) / np.maximum(tg[None, :], 1e-30)

        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, n))
        for i in range(n):
            ax.plot(time, ratio[:, i], color=colors[i], lw=1.3, alpha=0.9, label=sids[i])
        ax.axhline(1.0, color="#d62728", ls="--", lw=1.5, zorder=6, label="target")

        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Simulated / target,  $x_i(t)\,/\,\mu_i$")
        ax.set_title(f"{r.name}{suffix}: convergence to target ({n} species)")
        # Focus the y-view on the convergence band around 1 (the deep t=0 transient,
        # where ratio starts near 0, falls off the bottom).
        fr = ratio[-1, :]
        lo = min(fr.min(), 0.1)
        hi = max(fr.max(), 10.0)
        ax.set_ylim(lo / 2.0, hi * 2.0)
        # Legend only when it stays readable.
        if n <= 30:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=7, ncol=1 + n // 16, title="Species")
        else:
            ax.legend(loc="upper right", handles=[ax.lines[-1]])  # just the target line

        fig.savefig(out_root / f"{r.name}_simulation.png")
        plt.close(fig)


def plot_scaling(results: list[ScenarioResult], out_path: Path) -> None:
    """Total wall time and peak RAM vs input size — primary scaling figure."""
    sizes_mb = [r.input_size_bytes / 1024 / 1024 for r in results]
    totals = [sum(s.wall_s for s in r.stages) for r in results]
    peaks = [max((s.peak_rss_mb for s in r.stages), default=0.0) for r in results]
    labels = [r.name for r in results]

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ys, ylabel, color in (
        (ax_t, totals, "Total wall time (s)", "#1f77b4"),
        (ax_r, peaks, "Peak RSS (MB)", "#ff7f0e"),
    ):
        ax.plot(sizes_mb, ys, "o-", color=color, linewidth=2, markersize=8)
        for x, y, lbl in zip(sizes_mb, ys, labels):
            ax.annotate(f"{lbl}\n{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=9)
        ax.set_xlabel("Combined input size (MB)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    ax_t.set_title("Pipeline scaling — wall time")
    ax_r.set_title("Pipeline scaling — peak memory")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# Output dir housekeeping -----------------------------------------------------

def _clean_output_dir(out_root: Path) -> None:
    """Migrate any legacy subfolder artifacts to flat naming, then remove subfolders."""
    for sc in SCENARIOS:
        old_dir = out_root / sc["name"]
        if not old_dir.is_dir():
            continue
        # Migrate cached targets so we do not re-spend an LLM call needlessly.
        old_targets = old_dir / "targets.csv"
        new_targets = out_root / f"{sc['name']}_targets.csv"
        if old_targets.exists() and not new_targets.exists():
            shutil.move(str(old_targets), str(new_targets))
        shutil.rmtree(old_dir)


# Entry point -----------------------------------------------------------------

def load_results(out_root: Path) -> list[ScenarioResult]:
    """Reconstruct ScenarioResult objects from a previous run's saved artifacts
    (reports/results.json + per-scenario *_simulation.npz), so plots and reports
    can be regenerated without re-running the (expensive) pipeline."""
    json_path = out_root / "reports" / "results.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"No saved results at {json_path}; run the benchmark once before --replot."
        )
    data = json.loads(json_path.read_text())
    results: list[ScenarioResult] = []
    for d in data:
        stages = [StageMetric(**s) for s in d.pop("stages", [])]
        r = ScenarioResult(**d)
        r.stages = stages
        # Trajectory arrays live in the npz sidecar, not the JSON.
        npz = out_root / f"{r.name}_simulation.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            r.sim_time = z["time"]
            r.sim_trajectories = z["trajectories"]
            r.sim_species_ids = [str(s) for s in z["species_ids"]]
            r.sim_targets = z["targets"]
        results.append(r)
    return results


def write_outputs(results: list[ScenarioResult], out_root: Path) -> None:
    """Persist the raw results, then (re)generate every report and plot."""
    reports_dir = out_root / "reports"
    perf_dir = out_root / "plots" / "performance"
    opt_dir = out_root / "plots" / "optimization"
    sim_dir = out_root / "plots" / "simulation"
    for d in (reports_dir, perf_dir, opt_dir, sim_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Persist raw data first so a later plotting error never loses the run.
    (reports_dir / "results.json").write_text(
        json.dumps(
            [{**asdict(r), "stages": [asdict(s) for s in r.stages]} for r in results],
            indent=2,
            default=str,
        )
    )
    write_csv_report(results, reports_dir / "results.csv")
    write_markdown_summary(results, reports_dir / "summary.md")

    _grouped_bar(results, "wall_s", "Wall time (s, log scale)",
                 "Stage wall time per scenario", perf_dir / "time_per_stage.png",
                 logy=True, fmt="{:.2f}s")
    _grouped_bar(results, "peak_rss_mb", "Peak RSS (MB, log scale)",
                 "Stage peak memory per scenario", perf_dir / "ram_per_stage.png",
                 logy=True, fmt="{:.0f}")
    plot_scaling(results, perf_dir / "scaling.png")
    plot_loss_curves(results, opt_dir / "loss_curves.png")
    plot_final_errors(results, opt_dir / "final_errors.png")
    plot_simulation_trajectories(results, sim_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="bench_out", help="Flat output directory")
    ap.add_argument("--llm-model", default="llama3.1:8b", help="Ollama model for target generation")
    ap.add_argument("--sim-end", type=float, default=100000.0,
                    help="Simulation horizon used during optimization")
    ap.add_argument("--iterations", type=int, default=5000, help="OpenAI-ES iterations")
    ap.add_argument("--population-size", type=int, default=20,
                    help="OpenAI-ES population size")
    ap.add_argument("--learning-rate", type=float, default=0.01,
                    help="OpenAI-ES learning rate")
    ap.add_argument("--early-stop-loss", type=float, default=1e-4,
                    help="Stop optimization early once best loss drops to/below this "
                         "(set <=0 to disable and always run the full iteration budget)")
    ap.add_argument("--tol-rel", type=float, default=0.3,
                    help="Relative error tolerance for the simulation correctness check")
    ap.add_argument("--fallback-target", type=float, default=0.5,
                    help="Synthetic target value used when the LLM step fails")
    ap.add_argument("--skip-merge", action="store_true",
                    help="Skip merge and use file1 directly (useful for single inputs)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for ES population evaluation "
                         "(1 = serial; e.g. set to the number of CPU cores). Results "
                         "are identical to serial, only faster.")
    ap.add_argument("--replot", action="store_true",
                    help="Skip the pipeline; regenerate all reports and plots from a "
                         "previous run's saved data (reports/results.json + *_simulation.npz)")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    if args.replot:
        print(f"Reloading saved results from {out_root}/ and regenerating outputs...")
        results = load_results(out_root)
    else:
        _clean_output_dir(out_root)
        results = [run_scenario(sc, args, out_root) for sc in SCENARIOS]

    write_outputs(results, out_root)

    print(f"\n✓ Reports written under {out_root}/")
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_root)}")


if __name__ == "__main__":
    main()
