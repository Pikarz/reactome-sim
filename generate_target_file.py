from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import libsbml


@dataclass
class SpeciesInfo:
    species_id: str
    name: str
    compartment: str
    initial_amount: float | None
    initial_concentration: float | None
    boundary_condition: bool
    has_only_substance_units: bool
    constant: bool
    sbo_term: str
    metaid: str
    notes: str


def _xml_node_to_string(node: Any) -> str:
    if node is None:
        return ""
    try:
        return libsbml.writeXMLToString(node)
    except Exception:
        return ""


def _extract_notes(sbase: Any) -> str:
    try:
        if sbase is not None and sbase.isSetNotes():
            note_string = sbase.getNotesString()
            if note_string:
                return note_string
            return _xml_node_to_string(sbase.getNotes())
    except Exception:
        pass
    return ""


def _extract_annotation(sbase: Any) -> str:
    try:
        if sbase is not None and sbase.isSetAnnotation():
            annotation_string = sbase.getAnnotationString()
            if annotation_string:
                return annotation_string
            return _xml_node_to_string(sbase.getAnnotation())
    except Exception:
        pass
    return ""


def _clean_text(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_sbml(sbml_path: str) -> dict[str, Any]:
    reader = libsbml.SBMLReader()
    document = reader.readSBML(sbml_path)
    model = document.getModel()
    if model is None:
        errors = []
        for i in range(document.getNumErrors()):
            errors.append(document.getError(i).getMessage())
        raise ValueError("The SBML file does not contain a valid model.")

    model_info: dict[str, Any] = {
        "model_id": model.getId(),
        "model_name": model.getName(),
        "model_sbo_term": model.getSBOTermID() if model.isSetSBOTerm() else "",
        "model_notes": _extract_notes(model),
    }

    compartments = []
    for comp in model.getListOfCompartments():
        compartments.append(
            {
                "id": comp.getId(),
                "name": comp.getName(),
                "constant": bool(comp.getConstant()),
                "units": comp.getUnits() if comp.isSetUnits() else "",
                "sbo_term": comp.getSBOTermID() if comp.isSetSBOTerm() else "",
                "notes": _extract_notes(comp),
            }
        )

    '''
    parameters = []
    for param in model.getListOfParameters():
        parameters.append(
            {
                "id": param.getId(),
                "name": param.getName(),
                "value": param.getValue() if param.isSetValue() else None,
                "units": param.getUnits() if param.isSetUnits() else "",
                "constant": bool(param.getConstant()),
                "metaid": param.getMetaId(),
                "sbo_term": param.getSBOTermID() if param.isSetSBOTerm() else "",
            }
        )
    '''

    species_info: list[SpeciesInfo] = []
    for sp in model.getListOfSpecies():
        species_info.append(
            SpeciesInfo(
                species_id=sp.getId(),
                name=sp.getName(),
                compartment=sp.getCompartment(),
                initial_amount=sp.getInitialAmount() if sp.isSetInitialAmount() else None,
                initial_concentration=sp.getInitialConcentration() if sp.isSetInitialConcentration() else None,
                boundary_condition=bool(sp.getBoundaryCondition()),
                has_only_substance_units=bool(sp.getHasOnlySubstanceUnits()),
                constant=bool(sp.getConstant()),
                sbo_term=sp.getSBOTermID() if sp.isSetSBOTerm() else "",
                metaid=sp.getMetaId(),
                notes=_extract_notes(sp),
            )
        )

    reactions = []
    for rxn in model.getListOfReactions():
        reactants = [
            {
                "species": sr.getSpecies(),
                "stoichiometry": sr.getStoichiometry(),
                "constant": bool(sr.getConstant()),
                "id": sr.getId(),
            }
            for sr in rxn.getListOfReactants()
        ]
        products = [
            {
                "species": sr.getSpecies(),
                "stoichiometry": sr.getStoichiometry(),
                "constant": bool(sr.getConstant()),
                "id": sr.getId(),
            }
            for sr in rxn.getListOfProducts()
        ]
        modifiers = [m.getSpecies() for m in rxn.getListOfModifiers()]
        kinetic_law = None
        if rxn.isSetKineticLaw():
            kl = rxn.getKineticLaw()
            kinetic_law = {
                "math": _xml_node_to_string(kl.getMath()),
                "formula": kl.getFormula() if kl.isSetMath() else "",
                "local_parameters": [
                    {
                        "id": lp.getId(),
                        "value": lp.getValue() if lp.isSetValue() else None,
                        "units": lp.getUnits() if lp.isSetUnits() else "",
                    }
                    for lp in kl.getListOfLocalParameters()
                ],
            }

        reactions.append(
            {
                "id": rxn.getId(),
                "name": rxn.getName(),
                "metaid": rxn.getMetaId(),
                "compartment": rxn.getCompartment() if rxn.isSetCompartment() else "",
                "reversible": bool(rxn.getReversible()),
                "fast": bool(rxn.getFast()),
                "sbo_term": rxn.getSBOTermID() if rxn.isSetSBOTerm() else "",
                "notes": _extract_notes(rxn),
                "reactants": reactants,
                "products": products,
                "modifiers": modifiers,
                "kinetic_law": kinetic_law,
            }
        )

    '''
    rules = []
    for rule in model.getListOfRules():
        rule_type = "unknown"
        if rule.isAlgebraic():
            rule_type = "algebraic"
        elif rule.isAssignment():
            rule_type = "assignment"
        elif rule.isRate():
            rule_type = "rate"

        rules.append(
            {
                "type": rule_type,
                "variable": rule.getVariable() if hasattr(rule, "getVariable") else "",
                "formula": rule.getFormula(),
                "math": _xml_node_to_string(rule.getMath()),
                "metaid": rule.getMetaId(),
            }
        )
    '''

    '''
    constraints = []
    for cst in model.getListOfConstraints():
        constraints.append(
            {
                "metaid": cst.getMetaId(),
                "math": _xml_node_to_string(cst.getMath()),
                "message": _xml_node_to_string(cst.getMessage()),
            }
        )
    '''

    return {
        "model_info": model_info,
        "compartments": compartments,
        #"parameters": parameters,
        "species": [s.__dict__ for s in species_info],
        "reactions": reactions,
        #"rules": rules,
        #"constraints": constraints,
    }


def build_concentration_prompt(pathway_name: str, species_subset: list[dict[str, Any]]) -> str:
    # Ask only for per-species concentrations, for one chunk of species. Keeping the
    # request small (a few dozen species) keeps the JSON output within the model's
    # reliable generation length so it does not truncate into invalid JSON.
    table = [
        {
            "species_id": s["species_id"],
            "name": _clean_text(s.get("name", "")),
            "compartment": s.get("compartment", ""),
        }
        for s in species_subset
    ]
    prompt = f"""
        For pathway "{pathway_name}", provide a realistic MEAN steady-state CONCENTRATION
        (in mM, i.e. mmol/L) for each molecular species listed below.

        Output rules (mandatory):
        - Reply ONLY with valid JSON, no extra text.
        - Format:
        {{
        "targets": [
            {{"species_id": "<id>", "concentration": <number>}},
            ...
        ]
        }}
        - Include EXACTLY one entry for EACH species_id below.
        - concentration must be a finite real number >= 0 (mean concentration in mM).
        - Do not invent species ids that are not in the list.

        Species list (compact JSON):
        {json.dumps(table, ensure_ascii=False)}
    """
    return textwrap.dedent(prompt).strip()


def build_volume_prompt(pathway_name: str, compartments: list[dict[str, Any]]) -> str:
    # Ask only for compartment volumes (there are typically only a handful).
    table = [
        {"compartment_id": c.get("id", ""), "name": _clean_text(c.get("name", ""))}
        for c in compartments
    ]
    prompt = f"""
        For pathway "{pathway_name}", provide a realistic physical VOLUME (in litres, L)
        for each cellular compartment listed below.

        Output rules (mandatory):
        - Reply ONLY with valid JSON, no extra text.
        - Format:
        {{
        "compartments": [
            {{"compartment_id": "<id>", "volume_litres": <number>}},
            ...
        ]
        }}
        - Include EXACTLY one entry for EACH compartment_id below.
        - volume_litres must be a finite real number > 0 (compartment volume in L).
        - Do not invent ids that are not in the list.

        Compartment list (compact JSON):
        {json.dumps(table, ensure_ascii=False)}
    """
    return textwrap.dedent(prompt).strip()


def _pathway_name(sbml_data: dict[str, Any]) -> str:
    model = sbml_data["model_info"]
    return _clean_text(model.get("model_name", "")) or model.get("model_id", "unknown_pathway")


def _llm_request_with_retry(prompt: str, model: str, temperature: float, validate, retries: int = 3):
    # LLM output is stochastic: a chunk occasionally returns malformed JSON or a
    # missing id. Retry a few times before giving up so one bad response does not
    # abort the whole (multi-chunk) generation.
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            text = call_ollama(prompt, model, temperature)
            return validate(extract_json_from_text(text))
        except Exception as exc:  # malformed JSON, missing ids, etc.
            last_exc = exc
    raise RuntimeError(f"LLM request failed after {retries} attempts: {last_exc}")


def _parse_concentrations_lenient(text: str, allowed_ids: set[str]) -> dict[str, float]:
    # Extract whatever valid {species_id: concentration} pairs are present for the
    # allowed ids, ignoring malformed/missing/extra entries instead of raising. Used
    # by the chunked generator so a single imperfect response is not fatal.
    out: dict[str, float] = {}
    try:
        data = extract_json_from_text(text)
    except Exception:
        return out
    for item in data.get("targets", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        sid = item.get("species_id")
        val = item.get("concentration", item.get("target_value"))
        if sid in allowed_ids and val is not None:
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v >= 0:
                out[str(sid)] = v
    return out


def _request_concentrations(pathway, species_subset, model, temperature, attempts=2):
    # Ask the LLM for one chunk; tolerantly collect valid values across a couple of
    # attempts (each attempt may fill in ids the previous one missed).
    ids = {s["species_id"] for s in species_subset}
    got: dict[str, float] = {}
    for _ in range(attempts):
        prompt = build_concentration_prompt(pathway, species_subset)
        got.update(_parse_concentrations_lenient(call_ollama(prompt, model, temperature), ids))
        if ids.issubset(got):
            break
    return got


def generate_targets_chunked(
    sbml_data: dict[str, Any],
    model: str,
    temperature: float = 0.2,
    chunk_size: int = 30,
) -> tuple[dict[str, float], dict[str, float]]:
    """Query the LLM for concentrations in small species chunks (and volumes once),
    then merge. Large models otherwise emit a single oversized JSON that truncates
    into 'no targets list'. Tolerant: stragglers the LLM keeps omitting are retried,
    then filled with the median of the obtained values rather than aborting the run.
    Returns (concentration_by_species, volume_by_compartment)."""
    pathway = _pathway_name(sbml_data)
    species = sbml_data["species"]
    compartments = sbml_data["compartments"]

    # 1) Compartment volumes — a single small request (strict, with retry).
    volume_map = _llm_request_with_retry(
        build_volume_prompt(pathway, compartments), model, temperature,
        lambda js: validate_volumes(js, [c["id"] for c in compartments]),
    )

    # 2) Species concentrations — chunked, accumulated tolerantly.
    concentration_map: dict[str, float] = {}
    n_chunks = (len(species) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk = species[ci * chunk_size : (ci + 1) * chunk_size]
        concentration_map.update(_request_concentrations(pathway, chunk, model, temperature))
        print(f"  targets chunk {ci + 1}/{n_chunks}: {len(concentration_map)}/{len(species)} so far")

    # 3) One cleanup pass over any species still missing.
    missing = [s for s in species if s["species_id"] not in concentration_map]
    if missing:
        for ci in range(0, len(missing), chunk_size):
            concentration_map.update(
                _request_concentrations(pathway, missing[ci : ci + chunk_size], model, temperature)
            )

    # 4) Fill any remaining holes with the median so generation never aborts.
    still_missing = [s["species_id"] for s in species if s["species_id"] not in concentration_map]
    if still_missing:
        import statistics

        fallback = statistics.median(concentration_map.values()) if concentration_map else 0.5
        print(f"  WARNING: LLM omitted {len(still_missing)} species; filling with median={fallback:g}")
        for sid in still_missing:
            concentration_map[sid] = float(fallback)

    return concentration_map, volume_map


def call_ollama(prompt: str, model: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Network error while calling Ollama. Ensure Ollama is running on localhost:11434 "
            f"and the model is available (ollama pull {model}): {exc.reason}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Error while calling Ollama. Ensure Ollama is running and the model is available "
            f"(ollama pull {model})."
        ) from exc

    parsed = json.loads(body)
    text = str(parsed.get("message", {}).get("content", ""))

    if not text.strip():
        raise RuntimeError(f"Empty Ollama response: {body}")

    return text


def extract_json_from_text(text: str) -> dict[str, Any]:
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            raise ValueError("Could not find JSON in the model response.")
        return json.loads(match.group(0))


def validate_concentrations(data: dict[str, Any], expected_species_ids: list[str]) -> dict[str, float]:
    targets = data.get("targets")
    if not isinstance(targets, list):
        raise ValueError("The returned JSON does not contain a 'targets' list.")

    parsed: dict[str, float] = {}
    for item in targets:
        if not isinstance(item, dict):
            raise ValueError("Each item in 'targets' must be an object.")
        species_id = item.get("species_id")
        # Accept the new 'concentration' key, fall back to legacy 'target_value'.
        value = item.get("concentration", item.get("target_value"))

        if species_id is None:
            raise ValueError("Missing 'species_id' field in a 'targets' item.")
        if value is None:
            raise ValueError(f"Missing 'concentration' field for species_id={species_id}.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric concentration for species_id={species_id}: {value}") from exc

        if numeric_value < 0:
            raise ValueError(f"Negative concentration for species_id={species_id}: {numeric_value}")

        parsed[str(species_id)] = numeric_value

    missing = [sid for sid in expected_species_ids if sid not in parsed]
    if missing:
        raise ValueError(f"Missing species in LLM response: {missing}")

    return parsed


def validate_volumes(data: dict[str, Any], expected_compartment_ids: list[str]) -> dict[str, float]:
    comps = data.get("compartments")
    if not isinstance(comps, list):
        raise ValueError("The returned JSON does not contain a 'compartments' list.")

    parsed: dict[str, float] = {}
    for item in comps:
        if not isinstance(item, dict):
            raise ValueError("Each item in 'compartments' must be an object.")
        comp_id = item.get("compartment_id")
        volume = item.get("volume_litres", item.get("volume"))

        if comp_id is None:
            raise ValueError("Missing 'compartment_id' field in a 'compartments' item.")
        if volume is None:
            raise ValueError(f"Missing 'volume_litres' field for compartment_id={comp_id}.")

        try:
            numeric_value = float(volume)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric volume for compartment_id={comp_id}: {volume}") from exc

        if numeric_value <= 0:
            raise ValueError(f"Non-positive volume for compartment_id={comp_id}: {numeric_value}")

        parsed[str(comp_id)] = numeric_value

    missing = [cid for cid in expected_compartment_ids if cid not in parsed]
    if missing:
        raise ValueError(f"Missing compartments in LLM response: {missing}")

    return parsed


def build_target_rows(
    species: list[dict[str, Any]],
    concentration_by_species: dict[str, float],
    volume_by_compartment: dict[str, float],
) -> list[tuple[str, float, float, str, float]]:
    # We fit in CONCENTRATION: the optimization target is the species concentration
    # itself, not concentration * volume. The kinetic laws already operate on
    # concentrations, so fitting concentrations is the natural quantity and keeps the
    # ODE numerically stable. (Realistic femtoliter volumes would otherwise crush the
    # abundances to ~1e-30 and crash the integrator.) The compartment volume is still
    # recorded per row as provenance, so an abundance can be recovered downstream at
    # any time via abundance = concentration * volume.
    rows: list[tuple[str, float, float, str, float]] = []
    for s in species:
        sid = s["species_id"]
        if sid not in concentration_by_species:
            continue
        comp = s.get("compartment", "")
        volume = volume_by_compartment.get(comp, 1.0)
        concentration = concentration_by_species[sid]
        # target_value == concentration (volume kept only for provenance/recovery).
        rows.append((sid, concentration, concentration, comp, volume))
    return rows


def write_csv(rows: list[tuple[str, float, float, str, float]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # Column 2 (target_value) is the abundance the optimizer fits against;
        # the remaining columns preserve the concentration/volume provenance and
        # are consumed by the reference-injection step.
        writer.writerow(["species_id", "target_value", "concentration", "compartment", "volume"])
        for species_id, abundance, concentration, compartment, volume in rows:
            writer.writerow([species_id, abundance, concentration, compartment, volume])


def csv_has_reference_columns(csv_path: str) -> bool:
    # True only for the rich CSV that carries concentration + volume provenance.
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    return {"concentration", "compartment", "volume"}.issubset(set(header))


def read_reference_maps(csv_path: str) -> tuple[dict[str, float], dict[str, float]]:
    # Rebuild (concentration_by_species, volume_by_compartment) from the rich CSV
    # so the injection step does not need to re-query the LLM.
    if not csv_has_reference_columns(csv_path):
        raise ValueError(
            f"{csv_path} is missing concentration/compartment/volume columns "
            "(stale 2-column format). Regenerate targets so the references can be injected."
        )

    concentration_by_species: dict[str, float] = {}
    volume_by_compartment: dict[str, float] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = row.get("species_id")
            if not sid:
                continue
            concentration_by_species[sid] = float(row["concentration"])
            comp = row.get("compartment", "")
            if comp:
                volume_by_compartment[comp] = float(row["volume"])
    return concentration_by_species, volume_by_compartment


def resolve_output_paths(
    sbml_path: str,
    output_csv_arg: str | None,
    prompt_output_arg: str | None,
) -> tuple[str, str]:
    sbml_name = Path(sbml_path).stem
    output_dir = Path("generated_target") / sbml_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = str(Path(output_csv_arg)) if output_csv_arg else str(output_dir / "target.csv")
    prompt_output = (
        str(Path(prompt_output_arg))
        if prompt_output_arg
        else str(output_dir / f"prompt_{sbml_name}.txt")
    )
    return output_csv, prompt_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with average species targets from an SBML file using Ollama."
    )
    parser.add_argument("--sbml", required=True, help="Path to the .sbml file")
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path. If omitted, uses generated_target/<sbml_name>/target.csv",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model name (e.g., llama3.2:3b, gemma2:2b, mistral:7b)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument(
        "--prompt-output",
        default="",
        help="Prompt output path. If omitted, uses generated_target/<sbml_name>/prompt_<sbml_name>.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Does not call Ollama; generates only the prompt and exits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv_path, prompt_output_path = resolve_output_paths(
        sbml_path=args.sbml,
        output_csv_arg=args.output_csv,
        prompt_output_arg=args.prompt_output,
    )

    sbml_data = parse_sbml(args.sbml)
    pathway = _pathway_name(sbml_data)

    # Preview the prompts that would be sent (volume request + first species chunk).
    preview = (
        build_volume_prompt(pathway, sbml_data["compartments"])
        + "\n\n# --- example species chunk ---\n"
        + build_concentration_prompt(pathway, sbml_data["species"][:30])
    )
    with open(prompt_output_path, "w", encoding="utf-8") as handle:
        handle.write(preview)

    if args.dry_run:
        print(preview)
        print("\n[DRY RUN] Prompts generated. No API call was performed.")
        print(f"Prompt saved to: {prompt_output_path}")
        return 0

    # Chunked generation: concentrations a few dozen species at a time, volumes once.
    concentration_map, volume_map = generate_targets_chunked(
        sbml_data, args.model, temperature=args.temperature
    )
    rows = build_target_rows(sbml_data["species"], concentration_map, volume_map)

    write_csv(rows, output_csv_path)

    print(f"Prompt saved to: {prompt_output_path}")
    print(f"CSV generated: {output_csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

