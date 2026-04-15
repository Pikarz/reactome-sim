from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Iterable
import libsbml

_MATHML_NS_CANONICAL = "http://www.w3.org/1998/Math/MathML"
_MATHML_NS_ALIASES = (
    "http://www.w3.org/1998/math/MathML",
    "http://www.w3.org/1998/math/mathml",
    "http://www.w3.org/1998/Math/mathml",
)


def _safe_id(raw: str) -> str:
    # Normalize IDs so they are valid SBML identifiers.
    cleaned = []
    for ch in raw:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        else:
            cleaned.append("_")

    # Remove leading/trailing underscores created by replacement.
    out = "".join(cleaned).strip("_")

    # Ensure the ID is never empty.
    if not out:
        out = "id"

    # Ensure the ID does not start with a digit (invalid in SBML IDs).
    if out[0].isdigit():
        out = f"id_{out}"

    # Return the sanitized identifier.
    return out


def _species_token(species_id: str) -> str:
    # Match the hand-edited convention used in existing models.
    # Example: "species_2023929" becomes "2023929".
    if species_id.startswith("species_"):
        return species_id[len("species_"):]

    return _safe_id(species_id)


def _get_or_create_parameter(model: libsbml.Model, pid: str, value: float, constant: bool) -> libsbml.Parameter:
    # Reuse existing parameter when present to keep reruns deterministic.
    existing = model.getParameter(pid)
    if existing is not None:
        # Refresh value and constant flag so caller always gets the expected setup.
        existing.setValue(value)
        existing.setConstant(constant)
        return existing

    # Otherwise create the parameter from scratch.
    p = model.createParameter()
    p.setId(pid)
    p.setValue(value)
    p.setConstant(constant)
    return p


def _get_or_create_rate_rule(model: libsbml.Model, variable: str, formula: str) -> libsbml.RateRule:
    # Look for an existing rate rule over the same variable and update it in place.
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule is not None and rule.isRate() and rule.getVariable() == variable:
            rule.setMath(libsbml.parseL3Formula(formula))
            return rule

    # If no matching rule exists, create a new rate rule.
    rr = model.createRateRule()
    rr.setVariable(variable)
    rr.setMath(libsbml.parseL3Formula(formula))
    return rr


def _get_or_create_assignment_rule(model: libsbml.Model, variable: str, formula: str) -> libsbml.AssignmentRule:
    # Look for an existing assignment rule over the same variable and update it in place.
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule is not None and rule.isAssignment() and rule.getVariable() == variable:
            rule.setMath(libsbml.parseL3Formula(formula))
            return rule

    # If no matching rule exists, create a new assignment rule.
    ar = model.createAssignmentRule()
    ar.setVariable(variable)
    ar.setMath(libsbml.parseL3Formula(formula))
    return ar


def _has_mean_constraint(model: libsbml.Model, species_id: str) -> bool:
    # Mean constraint references species-specific symbol mu_<token>.
    token = f"mu_{_species_token(species_id)}"

    # Scan all constraints and detect whether that symbol is already present.
    for i in range(model.getNumConstraints()):
        c = model.getConstraint(i)
        if c is None or not c.isSetMath():
            continue
        text = libsbml.formulaToL3String(c.getMath())
        if token in text:
            return True

    # No matching mean constraint found.
    return False


def _reaction_has_kinetic_law(reaction: libsbml.Reaction) -> bool:
    # Reaction has valid kinetic law only if the kinetic law object exists and has math.
    return reaction.getKineticLaw() is not None and reaction.getKineticLaw().isSetMath()


def _pow_term(species_id: str, stoich: float) -> str:
    # For stoichiometry 1, omit exponent for cleaner formulas.
    if abs(stoich - 1.0) < 1e-12:
        return species_id

    # Integer exponents are rendered as integers for readability.
    if float(stoich).is_integer():
        return f"pow({species_id},{int(stoich)})"

    # Non-integer stoichiometry retains floating exponent.
    return f"pow({species_id},{stoich})"


def _build_mass_action_formula(reaction: libsbml.Reaction) -> str:
    # Collect multiplicative terms for all reactants in a standard mass-action product.
    terms: list[str] = []
    for sr in reaction.getListOfReactants():
        terms.append(_pow_term(sr.getSpecies(), sr.getStoichiometry()))

    # Keep spontaneous reactions disabled when reactants are missing.
    if not terms:
        return "0"

    # Join terms as a product expression.
    return " * ".join(terms)


def _add_kinetic_laws_if_missing(model: libsbml.Model) -> int:
    # Track how many reactions were updated.
    updated = 0

    # Fill only reactions missing kinetic laws; do not overwrite curated laws.
    for reaction in model.getListOfReactions():
        if _reaction_has_kinetic_law(reaction):
            continue
        kl = reaction.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(_build_mass_action_formula(reaction)))
        updated += 1

    # Return number of inserted kinetic laws for reporting.
    return updated


def _species_initial_value(species: libsbml.Species, fallback: float) -> float:
    # Prefer concentration when set.
    if species.isSetInitialConcentration():
        return float(species.getInitialConcentration())

    # Fall back to amount when concentration is not provided.
    if species.isSetInitialAmount():
        return float(species.getInitialAmount())

    # Final fallback for species without explicit initial state.
    return fallback


def _add_source_reaction(model: libsbml.Model, species: libsbml.Species, k_id: str) -> None:
    # Fixed ID used by the template for a synthetic source reaction.
    rid = "reaction_input"

    # If already present, keep existing reaction untouched.
    if model.getReaction(rid) is not None:
        return

    # Create an irreversible source reaction producing the selected species.
    rxn = model.createReaction()
    rxn.setId(rid)
    rxn.setReversible(False)
    rxn.setFast(False)

    # Keep compartment consistent with the target species when available.
    if species.isSetCompartment():
        rxn.setCompartment(species.getCompartment())

    # Add species as a product with unit stoichiometry.
    prod = rxn.createProduct()
    prod.setSpecies(species.getId())
    prod.setStoichiometry(1.0)
    prod.setConstant(True)

    # Source rate is controlled by parameter K_in (or whichever k_id is provided).
    kl = rxn.createKineticLaw()
    kl.setMath(libsbml.parseL3Formula(k_id))


def _add_sink_reaction(model: libsbml.Model, species: libsbml.Species, k_id: str) -> None:
    # Fixed ID used by the template for a synthetic first-order degradation reaction.
    rid = "reaction_output_degradation"

    # If already present, keep existing reaction untouched.
    if model.getReaction(rid) is not None:
        return

    # Create an irreversible sink reaction consuming the selected species.
    rxn = model.createReaction()
    rxn.setId(rid)
    rxn.setReversible(False)
    rxn.setFast(False)

    # Keep compartment consistent with the target species when available.
    if species.isSetCompartment():
        rxn.setCompartment(species.getCompartment())

    # Add species as a reactant with unit stoichiometry.
    rea = rxn.createReactant()
    rea.setSpecies(species.getId())
    rea.setStoichiometry(1.0)
    rea.setConstant(True)

    # Sink rate is proportional to species amount: K_out * species.
    kl = rxn.createKineticLaw()
    kl.setMath(libsbml.parseL3Formula(f"{k_id} * {species.getId()}"))


def _add_constraint(model: libsbml.Model, formula: str, message_text: str) -> None:
    # Create a numeric SBML constraint from a formula string.
    c = model.createConstraint()
    c.setMath(libsbml.parseL3Formula(formula))

    # `setMessage` is not always available across libSBML builds.
    if hasattr(c, "setMessage"):
        xml = (
            "<message>"
            f"<p xmlns='http://www.w3.org/1999/xhtml'>{message_text}</p>"
            "</message>"
        )
        node = libsbml.XMLNode.convertStringToXMLNode(xml)
        if node is not None:
            c.setMessage(node)


def _list_floating_species(model: libsbml.Model) -> Iterable[libsbml.Species]:
    # Iterate species that are dynamic state variables (not boundary, not constant).
    for sp in model.getListOfSpecies():
        if sp.getBoundaryCondition() or sp.getConstant():
            continue
        yield sp


def _remove_matching_parameters(model: libsbml.Model, predicate) -> None:
    # Delete backwards to keep indices stable.
    for i in range(model.getNumParameters() - 1, -1, -1):
        p = model.getParameter(i)
        if p is not None and predicate(p.getId()):
            model.removeParameter(i)


def _remove_matching_rules(model: libsbml.Model, predicate) -> None:
    # Delete backwards to keep indices stable.
    for i in range(model.getNumRules() - 1, -1, -1):
        r = model.getRule(i)

        # Not all rule types expose a variable name.
        var = r.getVariable() if r is not None and hasattr(r, "getVariable") else ""
        if r is not None and predicate(var):
            model.removeRule(i)


def _remove_matching_constraints(model: libsbml.Model, predicate) -> None:
    # Delete backwards to keep indices stable.
    for i in range(model.getNumConstraints() - 1, -1, -1):
        c = model.getConstraint(i)
        if c is None or not c.isSetMath():
            continue
        formula = libsbml.formulaToL3String(c.getMath())
        if predicate(formula):
            model.removeConstraint(i)


def _cleanup_previous_generated_content(model: libsbml.Model) -> None:
    # Clean previous generated artifacts so reruns stay idempotent.

    # Drop old generated parameter families from previous script versions.
    _remove_matching_parameters(
        model,
        lambda pid: (
            pid.startswith("z_")
            or pid.startswith("mu_species_")
            or pid.startswith("y_species_")
            or pid.startswith("y2_species_")
            or pid.startswith("K_in_species_")
            or pid.startswith("K_out_species_")
        ),
    )

    # Drop old generated rules tied to removed parameter families.
    _remove_matching_rules(
        model,
        lambda var: (
            var.startswith("z_")
            or var.startswith("y_species_")
            or var.startswith("y2_species_")
        ),
    )

    # Drop old constraints created by previous generator conventions.
    _remove_matching_constraints(
        model,
        lambda expr: ("z_" in expr or "mu_species_" in expr or "y_species_" in expr),
    )

    # Remove explicitly named legacy reaction(s), if present.
    for rid in (
        "reaction_input_clamping",
    ):
        if model.getReaction(rid) is not None:
            model.removeReaction(rid)

    # Remove old auto-generated source/sink reactions (legacy naming).
    for i in range(model.getNumReactions() - 1, -1, -1):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        rid = rxn.getId() or ""
        if rid.startswith("reaction_input_") or rid.startswith("reaction_output_"):
            model.removeReaction(i)


def _read_sbml_with_namespace_fix(input_path: Path) -> libsbml.SBMLDocument:
    # Some files use non-canonical MathML namespace casing; normalize before parsing.
    text = input_path.read_text(encoding="utf-8")

    # Replace each alias with canonical namespace to avoid parser inconsistencies.
    for alias in _MATHML_NS_ALIASES:
        text = text.replace(alias, _MATHML_NS_CANONICAL)

    # Parse normalized string into an SBML document object.
    return libsbml.readSBMLFromString(text)


def augment_model(model: libsbml.Model, default_mean: float, epsilon: float, threshold_m: float, k_default: float) -> dict[str, int]:
    # Keep summary metrics for CLI output and quick sanity checks.
    stats = {
        "kinetic_laws_added": 0,
        "source_reactions_added": 0,
        "sink_reactions_added": 0,
        "constraints_added": 0,
    }

    # Step 1: ensure every reaction has a kinetic law.
    stats["kinetic_laws_added"] = _add_kinetic_laws_if_missing(model)

    # Step 2: remove previously generated artifacts for idempotent reruns.
    _cleanup_previous_generated_content(model)

    # Step 3: create global numeric controls used in generated rules/constraints.
    _get_or_create_parameter(model, "epsilon", epsilon, True)
    _get_or_create_parameter(model, "M", threshold_m, True)

    # Step 4: create core tunable parameters (non-constant so optimizers can fit them).
    p_kin = _get_or_create_parameter(model, "K_in", 1.0, False)
    p_kout = _get_or_create_parameter(model, "K_out", 3.16, False)
    p_lam = _get_or_create_parameter(model, "lambda_1", 3.0, False)

    # Step 5: expose log-space versions to improve optimization stability and positivity.
    default_log_kin = math.log10(max(p_kin.getValue(), 1e-12))
    default_log_kout = math.log10(max(p_kout.getValue(), 1e-12))
    default_log_lam = math.log10(max(p_lam.getValue(), 1e-12))
    _get_or_create_parameter(model, "log_K_in", default_log_kin, False)
    _get_or_create_parameter(model, "log_K_out", default_log_kout, False)
    _get_or_create_parameter(model, "log_lambda_1", default_log_lam, False)

    # Step 6: tie linear-space parameters to log-space parameters through assignment rules.
    _get_or_create_assignment_rule(model, "K_in", "pow(10, log_K_in)")
    _get_or_create_assignment_rule(model, "K_out", "pow(10, log_K_out)")
    _get_or_create_assignment_rule(model, "lambda_1", "pow(10, log_lambda_1)")

    # Build producer/consumer maps so we can detect disconnected boundary needs per species.
    produced_by: dict[str, list[str]] = {}
    consumed_by: dict[str, list[str]] = {}
    for reaction in model.getListOfReactions():
        rid = reaction.getId()
        for sr in reaction.getListOfProducts():
            produced_by.setdefault(sr.getSpecies(), []).append(rid)
        for sr in reaction.getListOfReactants():
            consumed_by.setdefault(sr.getSpecies(), []).append(rid)

    # Step 7: augment each dynamic species with target, moments, and optional source/sink.
    for species in _list_floating_species(model):
        sid = species.getId()
        token = _species_token(sid)

        # Species-specific target mean and running moment state variables.
        _get_or_create_parameter(model, f"mu_{token}", default_mean, True)
        _get_or_create_parameter(model, f"y_{token}", 0.0, False)
        _get_or_create_parameter(model, f"y2_{token}", 0.0, False)

        # Running estimators for mean and second moment.
        _get_or_create_rate_rule(
            model,
            f"y_{token}",
            f"({sid} - y_{token}) / (time + epsilon)",
        )

        # Running second moment update rule: dy2/dt = (x^2 - y2)/(t + epsilon).
        _get_or_create_rate_rule(
            model,
            f"y2_{token}",
            f"(pow({sid},2) - y2_{token}) / (time + epsilon)",
        )

        # Determine whether this species is produced and/or consumed by any reaction.
        has_producers = sid in produced_by and len(produced_by[sid]) > 0
        has_consumers = sid in consumed_by and len(consumed_by[sid]) > 0

        # Add source/sink only if the species is disconnected on that side.
        if not has_producers:
            _add_source_reaction(model, species, "K_in")
            stats["source_reactions_added"] += 1

        # Add one global sink reaction when species lacks consumers.
        if not has_consumers:
            _add_sink_reaction(model, species, "K_out")
            stats["sink_reactions_added"] += 1

        # Enforce mean tracking target via absolute error inequality.
        if not _has_mean_constraint(model, sid):
            _add_constraint(
                model,
                f"abs(y_{token} - mu_{token}) <= 1e-3",
                f"Mean of {sid} must match mu_{token}",
            )
            stats["constraints_added"] += 1

    return stats


def validate_document(doc: libsbml.SBMLDocument) -> None:
    errors = []
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        if e.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            errors.append(e.getMessage())

    if errors:
        raise ValueError("SBML validation errors:\n- " + "\n- ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a Reactome SBML model with kinetic laws, input/output K parameters, rules, and constraints.",
    )

    parser.add_argument("--input", required=True, help="Input SBML file path")
    parser.add_argument("--output", help="Output SBML file path (default: <input>_augmented.sbml)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    parser.add_argument("--default-mean", type=float, default=0.5, help="Default mu_i target used for all species")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="epsilon for running mean/second moment rules")
    parser.add_argument("--threshold-m", type=float, default=1.0, help="Threshold M used by z_i rule")
    parser.add_argument("--k-default", type=float, default=0.1, help="Initial value for K_in/K_out parameters")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if args.inplace:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = in_path.with_name(f"{in_path.stem}_augmented{in_path.suffix}")

    doc = _read_sbml_with_namespace_fix(in_path)
    model = doc.getModel()
    if model is None:
        raise ValueError("Invalid SBML: missing <model>.")

    stats = augment_model(
        model,
        default_mean=args.default_mean,
        epsilon=args.epsilon,
        threshold_m=args.threshold_m,
        k_default=args.k_default,
    )

    # Validate produced SBML prior to writing.
    validate_document(doc)

    # Persist output document to disk.
    writer = libsbml.SBMLWriter()
    if not writer.writeSBMLToFile(doc, str(out_path)):
        raise RuntimeError(f"Failed to write output SBML to {out_path}")

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
