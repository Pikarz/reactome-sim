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


def build_prompt(sbml_data: dict[str, Any]) -> str:
    model = sbml_data["model_info"]
    species = sbml_data["species"]

    pathway_name = _clean_text(model.get("model_name", "")) or model.get("model_id", "unknown_pathway")
    species_list = ", ".join(s["species_id"] for s in species)

    context_json = json.dumps(sbml_data, ensure_ascii=False, indent=2)

    prompt = f"""
    For pathway {pathway_name}, provide the average values of the molecules in this pathway,
specifically: {species_list}.

You must use ALL information available in the SBML file (unique pathway identifiers,
RDF annotations, compartments, species, parameters, reactions, kinetic laws, rules,
constraints, notes, and metadata) provided below as JSON.

Goal:
Provide a realistic numeric target_value for each species_id.

Output rules (mandatory):
- Reply ONLY with valid JSON, with no extra text.
- Format:
    {{
    "pathway": "...",
    "targets": [
        {{"species_id": "species_x", "target_value": 0.123}},
        ...
    ]
    }}
- Include one item for EVERY species_id present.
- target_value must be a real number >= 0.
- Do not invent species_id values that are not in the context.

Full SBML context:
{context_json}
    """
    return textwrap.dedent(prompt).strip()


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


def validate_targets(data: dict[str, Any], expected_species_ids: list[str]) -> list[tuple[str, float]]:
    targets = data.get("targets")
    if not isinstance(targets, list):
        raise ValueError("The returned JSON does not contain a 'targets' list.")

    parsed_targets: dict[str, float] = {}
    for item in targets:
        if not isinstance(item, dict):
            raise ValueError("Each item in 'targets' must be an object.")
        species_id = item.get("species_id")
        target_value = item.get("target_value")

        if species_id is None:
            raise ValueError("Missing 'species_id' field in a 'targets' item.")
        if target_value is None:
            raise ValueError(f"Missing 'target_value' field for species_id={species_id}.")

        try:
            numeric_value = float(target_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric target_value for species_id={species_id}: {target_value}") from exc

        if numeric_value < 0:
            raise ValueError(f"Negative target_value for species_id={species_id}: {numeric_value}")

        parsed_targets[str(species_id)] = numeric_value

    missing = [sid for sid in expected_species_ids if sid not in parsed_targets]
    if missing:
        raise ValueError(f"Missing species in LLM response: {missing}")

    ordered = [(sid, parsed_targets[sid]) for sid in expected_species_ids]
    return ordered


def write_csv(rows: list[tuple[str, float]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species_id", "target_value"])
        for species_id, target_value in rows:
            writer.writerow([species_id, target_value])


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
    prompt = build_prompt(sbml_data)

    with open(prompt_output_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)

    if args.dry_run:
        print(prompt)
        print("\n[DRY RUN] Prompt generated. No API call was performed.")
        print(f"Prompt saved to: {prompt_output_path}")
        return 0

    model_response_text = call_ollama(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
    )
    response_json = extract_json_from_text(model_response_text)

    expected_species_ids = [s["species_id"] for s in sbml_data["species"]]
    rows = validate_targets(response_json, expected_species_ids)

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

