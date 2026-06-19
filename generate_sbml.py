from __future__ import annotations
import argparse
import math
import re
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
    return species_id


# Hill coefficient fixed at 10 per the model definition (steep, switch-like response).
_HILL_N = 10


def _hill_plus(species_id: str, m_id: str) -> str:
    # Activating Hill term H+(x, M) = (x/M)^n / (1 + (x/M)^n).
    # Near 0 when x << M, saturates to 1 when x >> M.
    ratio = f"pow({species_id} / {m_id}, {_HILL_N})"
    return f"({ratio} / (1 + {ratio}))"


def _hill_minus(species_id: str, m_id: str) -> str:
    # Inhibiting Hill term H-(x, M) = 1 / (1 + (x/M)^n).
    # Near 1 when x << M, drops toward 0 when x >> M.
    ratio = f"pow({species_id} / {m_id}, {_HILL_N})"
    return f"(1 / (1 + {ratio}))"


# SBO terms that denote a negative (inhibitory) modifier. Anything else acting as a
# modifier (catalyst, stimulator, generic modifier) is treated as positive.
_INHIBITOR_SBO = {
    "SBO:0000020",  # inhibitor
    "SBO:0000206",  # competitive inhibitor
    "SBO:0000207",  # non-competitive inhibitor
    "SBO:0000169",  # inhibition
    "SBO:0000536",  # partial inhibitor
    "SBO:0000537",  # complete inhibitor
}


def _modifier_is_inhibitor(modifier_ref: libsbml.ModifierSpeciesReference, species: libsbml.Species) -> bool:
    # Prefer the SBO on the modifier reference, fall back to the SBO on the species.
    sbo = ""
    if modifier_ref is not None and modifier_ref.isSetSBOTerm():
        sbo = modifier_ref.getSBOTermID()
    elif species is not None and species.isSetSBOTerm():
        sbo = species.getSBOTermID()
    return sbo in _INHIBITOR_SBO


def _modifier_hill_terms(model: libsbml.Model, reaction: libsbml.Reaction, threshold_m: float) -> list[str]:
    # One Hill factor per modifier: activators/catalysts gate the rate with H+ (the
    # reaction needs the regulator present), inhibitors gate it with H-. M is the
    # regulator's own per-species threshold (filled from references during injection).
    terms: list[str] = []
    for mr in reaction.getListOfModifiers():
        msid = mr.getSpecies()
        m_id = f"M_{_species_token(msid)}"
        # Ensure the threshold parameter exists even if the modifier is a boundary
        # species (which the per-species loop below would otherwise skip).
        _get_or_create_parameter(model, m_id, threshold_m, True)
        if _modifier_is_inhibitor(mr, model.getSpecies(msid)):
            terms.append(_hill_minus(msid, m_id))
        else:
            terms.append(_hill_plus(msid, m_id))
    return terms


def _build_mass_action_formula(reaction: libsbml.Reaction, k_id: str) -> str:
    # Collect multiplicative terms for all reactants in a standard mass-action product.
    terms: list[str] = []
    for sr in reaction.getListOfReactants():
        terms.append(_pow_term(sr.getSpecies(), sr.getStoichiometry()))

    # Keep spontaneous reactions disabled when reactants are missing.
    if not terms:
        return "0"

    # Join terms as a product expression.
    return f"{k_id} * " + " * ".join(terms)


def _add_kinetic_laws_if_missing(model: libsbml.Model, k_default: float, threshold_m: float) -> tuple[int, list[str]]:
    # Track how many reactions were updated.
    updated = 0
    tunable_params: list[str] = []
    log_default = math.log10(k_default) if k_default > 0 else 0.0

    # Fill only reactions missing kinetic laws; do not overwrite curated laws.
    for idx, reaction in enumerate(model.getListOfReactions()):
        if _reaction_has_kinetic_law(reaction):
            continue
        rid = reaction.getId() or f"reaction_{idx + 1}"
        token = _safe_id(rid)
        k_id = f"k_rxn_{token}"
        log_k_id = f"log_k_rxn_{token}"
        _get_or_create_parameter(model, k_id, k_default, False)
        _get_or_create_parameter(model, log_k_id, log_default, False)
        _get_or_create_assignment_rule(model, k_id, f"pow(10, {log_k_id})")
        # Mass-action base, optionally gated by Hill regulation from any modifiers.
        formula = _build_mass_action_formula(reaction, k_id)
        if formula != "0":
            hill_terms = _modifier_hill_terms(model, reaction, threshold_m)
            if hill_terms:
                formula = " * ".join([formula, *hill_terms])
        kl = reaction.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(formula))
        updated += 1
        tunable_params.append(log_k_id)

    # Return number of inserted kinetic laws for reporting.
    return updated, tunable_params


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
    # Per-species ID so every disconnected species gets its own source reaction.
    rid = f"reaction_input_{_species_token(species.getId())}"

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

    # Source rate is the constant influx K_in (paper form: dx/dt = K_in - consumers).
    kl = rxn.createKineticLaw()
    kl.setMath(libsbml.parseL3Formula(k_id))


def _add_sink_reaction(model: libsbml.Model, species: libsbml.Species, k_id: str) -> None:
    # Per-species ID so every disconnected species gets its own sink reaction.
    rid = f"reaction_output_degradation_{_species_token(species.getId())}"

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

    # Sink rate is proportional to the species amount: K_out * species
    # (paper form: dx/dt = producers - K_out * x).
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
            pid == "M"
            or pid.startswith("z_")
            or pid.startswith("M_")
            or pid.startswith("mu_species_")
            or pid.startswith("y_species_")
            or pid.startswith("y2_species_")
            or pid.startswith("K_in_species_")
            or pid.startswith("K_out_species_")
            or pid.startswith("k_rxn_")
            or pid.startswith("log_k_rxn_")
        ),
    )

    # Drop old generated rules tied to removed parameter families.
    _remove_matching_rules(
        model,
        lambda var: (
            var.startswith("z_")
            or var.startswith("y_species_")
            or var.startswith("y2_species_")
            or var.startswith("k_rxn_")
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


def _normalize_mathml_namespace(text: str) -> str:
    # Some Reactome/round-tripped files declare the MathML namespace under a prefix
    # (e.g. <ns7:math>, <ns8:apply>) and/or with non-canonical casing
    # (".../math/MathML"). libSBML's L3 validator rejects prefixed <math> blocks even
    # when the namespace URI is correct, so we rewrite them to default-namespace
    # MathML (<math xmlns="...Math/MathML">) before parsing.

    # 1) Canonicalize any alias casing of the MathML namespace URI.
    for alias in _MATHML_NS_ALIASES:
        text = text.replace(alias, _MATHML_NS_CANONICAL)

    # 2) Find every prefix currently bound to the (now canonical) MathML namespace.
    prefixes = set(
        re.findall(rf'xmlns:(\w+)="{re.escape(_MATHML_NS_CANONICAL)}"', text)
    )

    # 3) For each such prefix, strip it from element tags and put the MathML namespace
    #    as the default namespace on the <math> element itself.
    for prefix in prefixes:
        # Open tag of math: carry the namespace as default. Handles both
        # "<ns7:math>" and "<ns7:math attr=...>" forms.
        text = text.replace(f"<{prefix}:math", f'<math xmlns="{_MATHML_NS_CANONICAL}"')
        # Remaining open/self-closing tags (apply, ci, cn, power, ...): drop the prefix.
        text = text.replace(f"<{prefix}:", "<")
        # Closing tags: drop the prefix.
        text = text.replace(f"</{prefix}:", "</")
        # Remove the now-unused namespace declaration (with leading space if present).
        text = re.sub(rf'\s*xmlns:{prefix}="{re.escape(_MATHML_NS_CANONICAL)}"', "", text)

    return text


def _read_sbml_with_namespace_fix(input_path: Path) -> libsbml.SBMLDocument:
    # Normalize MathML namespace usage, then parse into an SBML document object.
    text = _normalize_mathml_namespace(input_path.read_text(encoding="utf-8"))
    return libsbml.readSBMLFromString(text)


def augment_model(model: libsbml.Model, default_mean: float, epsilon: float, threshold_m: float, k_default: float) -> dict:
    # Keep summary metrics for CLI output and quick sanity checks.
    stats = {
        "kinetic_laws_added": 0,
        "source_reactions_added": 0,
        "sink_reactions_added": 0,
        "constraints_added": 0,
        "tunable_params": [],
    }

    # Step 1: remove previously generated artifacts for idempotent reruns.
    _cleanup_previous_generated_content(model)

    # Step 2: ensure every reaction has a kinetic law.
    kinetic_laws_added, kinetic_tunables = _add_kinetic_laws_if_missing(model, k_default, threshold_m)
    stats["kinetic_laws_added"] = kinetic_laws_added
    stats["tunable_params"].extend(kinetic_tunables)

    # Step 3: create global numeric controls used in generated rules/constraints.
    # Hill thresholds M are now per-species (M_<token>), created in the loop below.
    _get_or_create_parameter(model, "epsilon", epsilon, True)

    # Build producer/consumer maps so we can detect disconnected boundary needs per species.
    produced_by: dict[str, list[str]] = {}
    consumed_by: dict[str, list[str]] = {}
    for reaction in model.getListOfReactions():
        rid = reaction.getId()
        for sr in reaction.getListOfProducts():
            produced_by.setdefault(sr.getSpecies(), []).append(rid)
        for sr in reaction.getListOfReactants():
            consumed_by.setdefault(sr.getSpecies(), []).append(rid)

    # Step 4: augment each dynamic species with target, moments, and optional per-species source/sink.
    for species in _list_floating_species(model):
        sid = species.getId()
        token = _species_token(sid)

        # Species-specific target mean and running moment state variables.
        _get_or_create_parameter(model, f"mu_{token}", default_mean, True)
        _get_or_create_parameter(model, f"y_{token}", 0.0, False)
        _get_or_create_parameter(model, f"y2_{token}", 0.0, False)

        # Per-species Hill threshold M_<token>. Placeholder until the reference
        # concentration from the LLM/literature is injected (see inject_references).
        m_id = f"M_{token}"
        _get_or_create_parameter(model, m_id, threshold_m, True)

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

        # Per-species source: independent K_in_<sid> so each disconnected species has its own production rate.
        if not has_producers:
            kin_id = f"K_in_{sid}"
            log_kin_id = f"log_K_in_{sid}"
            _get_or_create_parameter(model, kin_id, 1.0, False)
            _get_or_create_parameter(model, log_kin_id, 0.0, False)
            _get_or_create_assignment_rule(model, kin_id, f"pow(10, {log_kin_id})")
            _add_source_reaction(model, species, kin_id)
            stats["source_reactions_added"] += 1
            stats["tunable_params"].append(log_kin_id)

        # Per-species sink: independent K_out_<sid> so each disconnected species has its own degradation rate.
        if not has_consumers:
            kout_id = f"K_out_{sid}"
            log_kout_id = f"log_K_out_{sid}"
            _get_or_create_parameter(model, kout_id, 3.16, False)
            _get_or_create_parameter(model, log_kout_id, math.log10(3.16), False)
            _get_or_create_assignment_rule(model, kout_id, f"pow(10, {log_kout_id})")
            _add_sink_reaction(model, species, kout_id)
            stats["sink_reactions_added"] += 1
            stats["tunable_params"].append(log_kout_id)

        # Enforce mean tracking target via absolute error inequality.
        if not _has_mean_constraint(model, sid):
            _add_constraint(
                model,
                f"abs(y_{token} - mu_{token}) <= 1e-3",
                f"Mean of {sid} must match mu_{token}",
            )
            stats["constraints_added"] += 1

    return stats


def inject_references(
    model: libsbml.Model,
    concentration_by_species: dict[str, float],
    volume_by_compartment: dict[str, float],
    min_threshold: float = 1e-9,
) -> dict:
    # Write the reference concentrations (Hill thresholds M and target means mu) into
    # an already augmented model. Run between target generation and optimization.
    #
    # Compartment sizes are intentionally LEFT AT 1: we fit in concentration, not
    # abundance. Baking realistic femtoliter volumes into the compartment sizes would
    # scale species amounts down to ~1e-30 and make the ODE unintegrable. Volumes are
    # preserved in the targets CSV and can be applied as a post-hoc relabel
    # (abundance = concentration * volume) when abundances are needed for reporting.
    stats = {"m_set": 0, "mu_set": 0, "volumes_set": 0}

    # Per-species Hill threshold and target mean use the reference concentration.
    # The species symbol in the kinetic-law math is a concentration (the species is
    # not substance-only), so M must be expressed in concentration units too.
    for species in model.getListOfSpecies():
        sid = species.getId()
        if sid not in concentration_by_species:
            continue
        token = _species_token(sid)
        # Guard against a zero/negative threshold which would break the Hill ratio.
        m_value = max(float(concentration_by_species[sid]), min_threshold)

        m_param = model.getParameter(f"M_{token}")
        if m_param is not None:
            m_param.setValue(m_value)
            stats["m_set"] += 1

        mu_param = model.getParameter(f"mu_{token}")
        if mu_param is not None:
            mu_param.setValue(float(concentration_by_species[sid]))
            stats["mu_set"] += 1

    # Fit in concentration: pin every compartment size to 1 (not the real volume).
    # Setting it explicitly also guards against an unset/NaN size, which makes
    # RoadRunner produce NaN amounts and can crash the integrator. The real volume
    # stays in the targets CSV for post-hoc abundance recovery.
    for comp in model.getListOfCompartments():
        comp.setSize(1.0)
        comp.setConstant(True)
    _ = volume_by_compartment

    return stats


def inject_references_into_file(
    sbml_path: str,
    concentration_by_species: dict[str, float],
    volume_by_compartment: dict[str, float],
) -> dict:
    # Path-based wrapper: read, inject references, validate, write back in place.
    from pathlib import Path

    doc = _read_sbml_with_namespace_fix(Path(sbml_path))
    model = doc.getModel()
    if model is None:
        raise ValueError("Invalid SBML: missing <model>.")
    stats = inject_references(model, concentration_by_species, volume_by_compartment)
    validate_document(doc)
    writer = libsbml.SBMLWriter()
    if not writer.writeSBMLToFile(doc, str(sbml_path)):
        raise RuntimeError(f"Failed to write SBML to {sbml_path}")
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
    parser.add_argument("--threshold-m", type=float, default=1.0, help="Default per-species Hill threshold M (placeholder until references are injected)")
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
