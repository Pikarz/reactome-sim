from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import libsbml


LOGGER = logging.getLogger("sbml_merge")


@dataclass(frozen=True)
class MergePolicy:
    """Deterministic merge policy.

    prefer_file1=True means that when two matching entities conflict,
    the entity already present in the output model (originating from file1)
    is retained.
    """

    prefer_file1: bool = True


@dataclass
class EntityMergeSummary:
    merged: List[str] = field(default_factory=list)
    file1_as_is: List[str] = field(default_factory=list)
    file2_as_is: List[str] = field(default_factory=list)
    file1_modified: List[str] = field(default_factory=list)
    file2_modified: List[str] = field(default_factory=list)


@dataclass
class MergeRunSummary:
    compartments: EntityMergeSummary
    species: EntityMergeSummary
    parameters: EntityMergeSummary
    reactions: EntityMergeSummary


@dataclass
class _EntityTracker:
    file1_all: Set[str] = field(default_factory=set)
    merged: Set[str] = field(default_factory=set)
    file2_as_is: Set[str] = field(default_factory=set)
    file1_modified: Set[str] = field(default_factory=set)
    file2_modified: Set[str] = field(default_factory=set)

    def finalize(self) -> EntityMergeSummary:
        # Everything originally in file1 that was not merged and not edited in-place
        # is classified as "file1_as_is".
        file1_as_is = self.file1_all - self.merged - self.file1_modified
        return EntityMergeSummary(
            merged=sorted(self.merged),
            file1_as_is=sorted(file1_as_is),
            file2_as_is=sorted(self.file2_as_is),
            file1_modified=sorted(self.file1_modified),
            file2_modified=sorted(self.file2_modified),
        )


def _id_list(items: Iterable[libsbml.SBase]) -> Set[str]:
    out: Set[str] = set()
    for item in items:
        if item is not None and item.isSetId():
            out.add(item.getId())
    return out


def _configure_default_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _as_formula(math_node: Optional[libsbml.ASTNode]) -> str:
    if math_node is None:
        return ""
    return libsbml.formulaToL3String(math_node) or ""


def _safe_token(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "x"


def _make_unique_id(model: libsbml.Model, base_id: str) -> str:
    candidate = base_id
    idx = 2
    while (
        model.getElementBySId(candidate) is not None
        or model.getElementByMetaId(candidate) is not None
    ):
        candidate = f"{base_id}__{idx}"
        idx += 1
    return candidate


def _clone_document(doc: libsbml.SBMLDocument) -> libsbml.SBMLDocument:
    text = libsbml.writeSBMLToString(doc)
    clone = libsbml.readSBMLFromString(text)
    if clone is None or clone.getModel() is None:
        raise RuntimeError("Failed to clone SBML document")
    return clone


def _read_document(path: str) -> libsbml.SBMLDocument:
    doc = libsbml.readSBML(path)
    if doc is None:
        raise ValueError(f"Could not read SBML file: {path}")

    # Source models can contain package-specific consistency issues (for example,
    # Layout annotations) but still be parseable and mergeable. Only fatal read
    # failures stop processing; lower-severity issues are logged.
    if doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL) > 0:
        errors = []
        for i in range(min(10, doc.getNumErrors())):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_FATAL:
                errors.append(err.getMessage())
        raise ValueError(
            f"Input SBML has fatal parse errors in {path}: " + " | ".join(errors)
        )

    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        # Non-fatal source issues are logged because many real-world SBML files
        # are still mergeable even with package-specific warnings/errors.
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            LOGGER.warning("Input SBML issue in %s: %s", path, err.getMessage())

    if doc.getModel() is None:
        raise ValueError(f"Input SBML has no model: {path}")
    return doc


def _species_key(species_id: str, compartment_id: str) -> Tuple[str, str]:
    return (species_id, compartment_id)


def _species_compartment(species: libsbml.Species) -> str:
    if species is None or not species.isSetCompartment():
        return ""
    return species.getCompartment()


def _species_initial_numeric(species: libsbml.Species) -> float:
    if species is not None and species.isSetInitialConcentration():
        return float(species.getInitialConcentration())
    if species is not None and species.isSetInitialAmount():
        return float(species.getInitialAmount())
    return 0.0


def _species_initial_representation(species: libsbml.Species) -> str:
    if species is not None and species.isSetInitialConcentration():
        return "concentration"
    if species is not None and species.isSetInitialAmount():
        return "amount"
    return "missing"


def _ensure_species_initial(species: libsbml.Species) -> bool:
    """Ensure a species has an explicit initial value.

    Returns True when a missing initial value was filled with 0.
    """

    if species is None:
        return False
    if species.isSetInitialConcentration() or species.isSetInitialAmount():
        return False
    species.setInitialAmount(0.0)
    return True


def _normalize_stoich(value: float) -> float:
    return float(f"{value:.12g}")


def _resolve_model_substance_units(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    logger: logging.Logger = LOGGER,
) -> None:
    """Ensure model-level substanceUnits is set deterministically.

    Priority order:
    1) keep file1 model value if already set in merged_model
    2) use file2 model value if available
    3) fallback to built-in unit "mole"
    """

    if merged_model is None:
        return

    if merged_model.isSetSubstanceUnits():
        logger.info(
            "Keeping model substanceUnits from file1: %s",
            merged_model.getSubstanceUnits(),
        )
        return

    if model2 is not None and model2.isSetSubstanceUnits():
        val = model2.getSubstanceUnits()
        merged_model.setSubstanceUnits(val)
        logger.info("Set model substanceUnits from file2: %s", val)
        return

    merged_model.setSubstanceUnits("mole")
    logger.info(
        "Model substanceUnits missing in both inputs. Applied deterministic fallback: mole"
    )


def _reaction_side_signature(
    refs: Iterable[libsbml.SpeciesReference],
) -> Tuple[Tuple[str, float, bool], ...]:
    vals: List[Tuple[str, float, bool]] = []
    for ref in refs:
        sid = ref.getSpecies() if ref is not None and ref.isSetSpecies() else ""
        stoich = ref.getStoichiometry() if ref is not None else 1.0
        const = ref.getConstant() if ref is not None else True
        vals.append((sid, _normalize_stoich(stoich), bool(const)))
    return tuple(sorted(vals))


def _reaction_mod_signature(
    refs: Iterable[libsbml.ModifierSpeciesReference],
) -> Tuple[str, ...]:
    vals: List[str] = []
    for ref in refs:
        sid = ref.getSpecies() if ref is not None and ref.isSetSpecies() else ""
        vals.append(sid)
    return tuple(sorted(vals))


def _reaction_context(model: libsbml.Model, reaction: libsbml.Reaction) -> Tuple[str, ...]:
    compartments = set()
    if reaction is not None and reaction.isSetCompartment():
        compartments.add(reaction.getCompartment())

    def _add_comp(ref: libsbml.SimpleSpeciesReference) -> None:
        if ref is None or not ref.isSetSpecies():
            return
        sp = model.getSpecies(ref.getSpecies())
        if sp is None:
            return
        comp = _species_compartment(sp)
        if comp:
            compartments.add(comp)

    for rr in reaction.getListOfReactants():
        _add_comp(rr)
    for pr in reaction.getListOfProducts():
        _add_comp(pr)
    for mr in reaction.getListOfModifiers():
        _add_comp(mr)

    return tuple(sorted(compartments))


def _reaction_signature(model: libsbml.Model, reaction: libsbml.Reaction) -> Tuple:
    kl = reaction.getKineticLaw()
    kl_formula = _as_formula(kl.getMath()) if kl is not None and kl.isSetMath() else ""
    return (
        reaction.getId(),
        _reaction_context(model, reaction),
        _reaction_side_signature(reaction.getListOfReactants()),
        _reaction_side_signature(reaction.getListOfProducts()),
        _reaction_mod_signature(reaction.getListOfModifiers()),
        bool(reaction.getReversible()),
        kl_formula,
    )


def _rename_species_references(
    sbase: libsbml.SBase, species_id_map: Dict[str, str]
) -> None:
    # renameSIdRefs updates formula references and direct symbol references.
    for old_id in sorted(species_id_map):
        new_id = species_id_map[old_id]
        if old_id == new_id:
            continue
        sbase.renameSIdRefs(old_id, new_id)


def merge_compartments(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    tracker: Optional[_EntityTracker] = None,
    logger: logging.Logger = LOGGER,
) -> Dict[str, str]:
    """Merge compartments by exact id.

    Compartments with matching ids are treated as the same entity.
    Compartments with different ids are always distinct.
    """

    compartment_map: Dict[str, str] = {}
    compartments = sorted(
        list(model2.getListOfCompartments()),
        key=lambda c: c.getId() if c is not None and c.isSetId() else "",
    )

    for comp2 in compartments:
        if comp2 is None or not comp2.isSetId():
            logger.warning("Skipping compartment without id from file2")
            continue

        cid = comp2.getId()
        compartment_map[cid] = cid
        comp1 = merged_model.getCompartment(cid)

        if comp1 is None:
            # Compartment exists only in file2: copy it unchanged.
            rc = merged_model.addCompartment(comp2.clone())
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(f"Failed to add compartment {cid}, libSBML code={rc}")
            if tracker is not None:
                tracker.file2_as_is.add(cid)
            continue

        # Same id in both files: this is considered a merged entity.
        if tracker is not None:
            tracker.merged.add(cid)

        conflicts = []
        if comp1.isSetName() and comp2.isSetName() and comp1.getName() != comp2.getName():
            conflicts.append(f"name: '{comp1.getName()}' vs '{comp2.getName()}'")
        if comp1.isSetSize() and comp2.isSetSize() and comp1.getSize() != comp2.getSize():
            conflicts.append(f"size: {comp1.getSize()} vs {comp2.getSize()}")
        if conflicts:
            logger.warning(
                "Compartment conflict for id '%s'. Keeping file1 values. Differences: %s",
                cid,
                "; ".join(conflicts),
            )

    return compartment_map


def merge_species(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    compartment_map: Dict[str, str],
    tracker: Optional[_EntityTracker] = None,
    logger: logging.Logger = LOGGER,
) -> Dict[str, str]:
    """Merge species by (id, compartment).

    If file2 species id collides with file1 species id but compartment differs,
    file2 species is retained with a deterministic renamed id and all references
    are remapped later.
    """

    species_id_map: Dict[str, str] = {}
    file1_species_ids = _id_list(merged_model.getListOfSpecies())
    merged_species_ids: Set[str] = set()
    file2_added_ids: Set[str] = set()
    species2 = sorted(
        list(model2.getListOfSpecies()),
        key=lambda s: s.getId() if s is not None and s.isSetId() else "",
    )

    for sp2 in species2:
        if sp2 is None or not sp2.isSetId():
            logger.warning("Skipping species without id from file2")
            continue

        sid2 = sp2.getId()
        comp2_raw = _species_compartment(sp2)
        comp2 = compartment_map.get(comp2_raw, comp2_raw)
        sp1 = merged_model.getSpecies(sid2)

        if sp1 is None:
            # Species exists only in file2: import it, normalizing compartment id
            # and guaranteeing an explicit initial value.
            clone = sp2.clone()
            changed = False
            if comp2 and clone.isSetCompartment() and clone.getCompartment() != comp2:
                clone.setCompartment(comp2)
                changed = True
            # Species present only in file2 keeps its own initial value when
            # available; otherwise default to 0.
            if _ensure_species_initial(clone):
                changed = True
            rc = merged_model.addSpecies(clone)
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(f"Failed to add species {sid2}, libSBML code={rc}")
            new_sid = clone.getId() if clone.isSetId() else sid2
            species_id_map[sid2] = new_sid
            file2_added_ids.add(new_sid)
            if tracker is not None:
                if changed:
                    # Imported from file2 but adjusted by merge policy.
                    tracker.file2_modified.add(new_sid)
                else:
                    # Imported from file2 exactly as-is.
                    tracker.file2_as_is.add(new_sid)
            continue

        comp1 = _species_compartment(sp1)
        if _species_key(sid2, comp1) == _species_key(sid2, comp2):
            # Same id + same compartment => merge.
            # Initial value policy: deterministic average of file1/file2 values,
            # interpreting missing values as 0.
            species_id_map[sid2] = sid2
            merged_species_ids.add(sid2)
            if tracker is not None:
                tracker.merged.add(sid2)
            rep1 = _species_initial_representation(sp1)
            rep2 = _species_initial_representation(sp2)
            val1 = _species_initial_numeric(sp1)
            val2 = _species_initial_numeric(sp2)
            avg = 0.5 * (val1 + val2)

            target_rep = "concentration" if (rep1 == "concentration" or rep2 == "concentration") else "amount"

            if target_rep == "concentration":
                if hasattr(sp1, "unsetInitialAmount"):
                    sp1.unsetInitialAmount()
                sp1.setInitialConcentration(avg)
            else:
                if hasattr(sp1, "unsetInitialConcentration"):
                    sp1.unsetInitialConcentration()
                sp1.setInitialAmount(avg)

            if val1 != val2 or rep1 == "missing" or rep2 == "missing":
                logger.info(
                    "Species initial value merged for '%s' in compartment '%s': "
                    "file1=%s (%s), file2=%s (%s), avg=%s (%s)",
                    sid2,
                    comp2,
                    val1,
                    rep1,
                    val2,
                    rep2,
                    avg,
                    target_rep,
                )
            continue

        # Same id but different compartments must remain separate.
        # We keep both entities by deterministically renaming the file2 species.
        suffix = _safe_token(comp2 if comp2 else "no_compartment")
        candidate = f"{sid2}__{suffix}"
        new_sid = _make_unique_id(merged_model, candidate)
        clone = sp2.clone()
        clone.setId(new_sid)
        if comp2 and clone.isSetCompartment() and clone.getCompartment() != comp2:
            clone.setCompartment(comp2)
        rc = merged_model.addSpecies(clone)
        if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(
                f"Failed to add renamed species {new_sid}, libSBML code={rc}"
            )
        species_id_map[sid2] = new_sid
        file2_added_ids.add(new_sid)
        if tracker is not None:
            tracker.file2_modified.add(f"{sid2}->{new_sid}")
        logger.warning(
            "Species id collision with different compartments for '%s': '%s' vs '%s'. "
            "Retained both by renaming file2 species to '%s'.",
            sid2,
            comp1,
            comp2,
            new_sid,
        )

    # Species present only in file1 are already in merged_model because merged
    # starts from file1. Keep their existing initial value; if missing, set 0.
    filled_missing = 0
    for sp in merged_model.getListOfSpecies():
        if _ensure_species_initial(sp):
            # Late normalization pass: catches all species that still lack an
            # explicit initial value after merge operations.
            filled_missing += 1
            sid = sp.getId() if sp is not None and sp.isSetId() else ""
            if tracker is not None and sid:
                if sid in file1_species_ids and sid not in merged_species_ids:
                    # Existing file1 species edited in-place.
                    tracker.file1_modified.add(sid)
                elif sid in file2_added_ids:
                    # Imported file2 species that had to be normalized.
                    if sid in tracker.file2_as_is:
                        tracker.file2_as_is.remove(sid)
                    tracker.file2_modified.add(sid)

    if filled_missing > 0:
        logger.info(
            "Filled missing initial values with 0 for %s species present in a single model.",
            filled_missing,
        )

    return species_id_map


def merge_parameters(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    policy: MergePolicy = MergePolicy(),
    tracker: Optional[_EntityTracker] = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Merge parameters by id with deterministic conflict resolution."""

    params2 = sorted(
        list(model2.getListOfParameters()),
        key=lambda p: p.getId() if p is not None and p.isSetId() else "",
    )

    for p2 in params2:
        if p2 is None or not p2.isSetId():
            logger.warning("Skipping parameter without id from file2")
            continue

        pid = p2.getId()
        p1 = merged_model.getParameter(pid)
        if p1 is None:
            # Parameter exists only in file2: copy unchanged.
            rc = merged_model.addParameter(p2.clone())
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(f"Failed to add parameter {pid}, libSBML code={rc}")
            if tracker is not None:
                tracker.file2_as_is.add(pid)
            continue

        # Same id in both files => merged parameter (file1-preferred on conflict).
        if tracker is not None:
            tracker.merged.add(pid)

        if p1.isSetValue() and p2.isSetValue() and p1.getValue() != p2.getValue():
            winner = "file1" if policy.prefer_file1 else "file2"
            logger.warning(
                "Parameter value conflict for '%s': file1=%s, file2=%s. "
                "Deterministic resolution: keeping %s.",
                pid,
                p1.getValue(),
                p2.getValue(),
                winner,
            )
            if not policy.prefer_file1:
                p1.setValue(p2.getValue())


def merge_reactions(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    species_id_map: Dict[str, str],
    policy: MergePolicy = MergePolicy(),
    tracker: Optional[_EntityTracker] = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Merge reactions by id and inferred compartment context from species.

    Reactions are merged only when id and compartment context match.
    On conflict, file1 reaction is kept by default.
    """

    reactions2 = sorted(
        list(model2.getListOfReactions()),
        key=lambda r: r.getId() if r is not None and r.isSetId() else "",
    )

    for r2 in reactions2:
        if r2 is None or not r2.isSetId():
            logger.warning("Skipping reaction without id from file2")
            continue

        rid = r2.getId()
        ref_renamed = False
        # Pre-check if reaction references symbols that will be renamed due to
        # species id collisions; this helps classify file2_as_is vs file2_modified.
        for ref in r2.getListOfReactants():
            if ref is not None and ref.isSetSpecies():
                sid = ref.getSpecies()
                if sid in species_id_map and species_id_map[sid] != sid:
                    ref_renamed = True
                    break
        if not ref_renamed:
            for ref in r2.getListOfProducts():
                if ref is not None and ref.isSetSpecies():
                    sid = ref.getSpecies()
                    if sid in species_id_map and species_id_map[sid] != sid:
                        ref_renamed = True
                        break
        if not ref_renamed:
            for ref in r2.getListOfModifiers():
                if ref is not None and ref.isSetSpecies():
                    sid = ref.getSpecies()
                    if sid in species_id_map and species_id_map[sid] != sid:
                        ref_renamed = True
                        break

        clone = r2.clone()
        # Rewrite all symbol references to keep the merged graph consistent.
        _rename_species_references(clone, species_id_map)

        # Ensure direct species refs are consistent even if renameSIdRefs does not
        # touch every reference type in some SBML builds.
        for rr in clone.getListOfReactants():
            sid = rr.getSpecies() if rr.isSetSpecies() else ""
            if sid in species_id_map:
                rr.setSpecies(species_id_map[sid])
        for pr in clone.getListOfProducts():
            sid = pr.getSpecies() if pr.isSetSpecies() else ""
            if sid in species_id_map:
                pr.setSpecies(species_id_map[sid])
        for mr in clone.getListOfModifiers():
            sid = mr.getSpecies() if mr.isSetSpecies() else ""
            if sid in species_id_map:
                mr.setSpecies(species_id_map[sid])

        r1 = merged_model.getReaction(rid)
        if r1 is None:
            # Reaction exists only in file2: add it directly (possibly with remapped refs).
            rc = merged_model.addReaction(clone)
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(f"Failed to add reaction {rid}, libSBML code={rc}")
            if tracker is not None:
                if ref_renamed:
                    tracker.file2_modified.add(rid)
                else:
                    tracker.file2_as_is.add(rid)
            continue

        ctx1 = _reaction_context(merged_model, r1)
        ctx2 = _reaction_context(merged_model, clone)

        if ctx1 == ctx2:
            # Same reaction id and same inferred biological context: treat as merge.
            if tracker is not None:
                tracker.merged.add(rid)
            sig1 = _reaction_signature(merged_model, r1)
            sig2 = _reaction_signature(merged_model, clone)

            if sig1 == sig2:
                continue

            winner = "file1" if policy.prefer_file1 else "file2"
            logger.warning(
                "Reaction conflict for id '%s' in context %s. "
                "Deterministic resolution: keeping %s reaction.",
                rid,
                ctx1,
                winner,
            )
            if not policy.prefer_file1:
                merged_model.removeReaction(rid)
                rc = merged_model.addReaction(clone)
                if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                    raise RuntimeError(
                        f"Failed to replace reaction {rid}, libSBML code={rc}"
                    )
            continue

        # Same id but different contexts: retain both with deterministic rename.
        context_token = _safe_token("_".join(ctx2) if ctx2 else "unknown_context")
        new_rid = _make_unique_id(merged_model, f"{rid}__{context_token}")
        clone.setId(new_rid)
        rc = merged_model.addReaction(clone)
        if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(
                f"Failed to add renamed reaction {new_rid}, libSBML code={rc}"
            )
        if tracker is not None:
            tracker.file2_modified.add(f"{rid}->{new_rid}")
        logger.warning(
            "Reaction id collision with different context for '%s': %s vs %s. "
            "Retained both by renaming file2 reaction to '%s'.",
            rid,
            ctx1,
            ctx2,
            new_rid,
        )


def _merge_unit_definitions(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    logger: logging.Logger = LOGGER,
) -> None:
    udefs2 = sorted(
        list(model2.getListOfUnitDefinitions()),
        key=lambda u: u.getId() if u is not None and u.isSetId() else "",
    )
    for u2 in udefs2:
        if u2 is None or not u2.isSetId():
            logger.warning("Skipping unit definition without id from file2")
            continue
        uid = u2.getId()
        u1 = merged_model.getUnitDefinition(uid)
        if u1 is None:
            rc = merged_model.addUnitDefinition(u2.clone())
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(
                    f"Failed to add unit definition {uid}, libSBML code={rc}"
                )
            continue
        if u1.toSBML() != u2.toSBML():
            logger.warning(
                "Unit definition conflict for '%s'. Keeping file1 definition.", uid
            )


def _merge_function_definitions(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    logger: logging.Logger = LOGGER,
) -> None:
    fdefs2 = sorted(
        list(model2.getListOfFunctionDefinitions()),
        key=lambda f: f.getId() if f is not None and f.isSetId() else "",
    )
    for f2 in fdefs2:
        if f2 is None or not f2.isSetId():
            logger.warning("Skipping function definition without id from file2")
            continue
        fid = f2.getId()
        f1 = merged_model.getFunctionDefinition(fid)
        if f1 is None:
            rc = merged_model.addFunctionDefinition(f2.clone())
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(
                    f"Failed to add function definition {fid}, libSBML code={rc}"
                )
            continue
        if _as_formula(f1.getMath()) != _as_formula(f2.getMath()):
            logger.warning(
                "Function definition conflict for '%s'. Keeping file1 definition.",
                fid,
            )


def _rule_key(rule: libsbml.Rule) -> Tuple[str, str]:
    rule_type = "unknown"
    variable = ""
    if rule is not None:
        if rule.isAssignment():
            rule_type = "assignment"
        elif rule.isRate():
            rule_type = "rate"
        elif rule.isAlgebraic():
            rule_type = "algebraic"
        if hasattr(rule, "isSetVariable") and rule.isSetVariable():
            variable = rule.getVariable()

    if variable:
        return (rule_type, variable)
    # Algebraic rules have no id/variable; use normalized math as deterministic key.
    return (rule_type, _as_formula(rule.getMath() if rule is not None else None))


def _merge_rules(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    species_id_map: Dict[str, str],
    logger: logging.Logger = LOGGER,
) -> None:
    existing = {_rule_key(r): r for r in merged_model.getListOfRules()}
    rules2 = list(model2.getListOfRules())
    rules2.sort(key=lambda r: _rule_key(r))

    for r2 in rules2:
        key = _rule_key(r2)
        r1 = existing.get(key)
        if r1 is None:
            clone = r2.clone()
            _rename_species_references(clone, species_id_map)
            rc = merged_model.addRule(clone)
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(f"Failed to add rule {key}, libSBML code={rc}")
            existing[key] = clone
            continue

        if _as_formula(r1.getMath()) != _as_formula(r2.getMath()):
            logger.warning("Rule conflict for key %s. Keeping file1 rule.", key)


def _event_key(event: libsbml.Event) -> Tuple[str, str, str]:
    eid = event.getId() if event is not None and event.isSetId() else ""
    trigger = ""
    delay = ""
    if event is not None and event.isSetTrigger() and event.getTrigger().isSetMath():
        trigger = _as_formula(event.getTrigger().getMath())
    if event is not None and event.isSetDelay() and event.getDelay().isSetMath():
        delay = _as_formula(event.getDelay().getMath())
    if eid:
        return ("id", eid, "")
    return ("content", trigger, delay)


def _event_signature(event: libsbml.Event) -> Tuple:
    trigger = ""
    delay = ""
    priority = ""

    if event.isSetTrigger() and event.getTrigger().isSetMath():
        trigger = _as_formula(event.getTrigger().getMath())
    if event.isSetDelay() and event.getDelay().isSetMath():
        delay = _as_formula(event.getDelay().getMath())
    if event.isSetPriority() and event.getPriority().isSetMath():
        priority = _as_formula(event.getPriority().getMath())

    assignments = []
    for ea in event.getListOfEventAssignments():
        var = ea.getVariable() if ea.isSetVariable() else ""
        formula = _as_formula(ea.getMath() if ea.isSetMath() else None)
        assignments.append((var, formula))

    return (
        trigger,
        delay,
        priority,
        bool(event.getUseValuesFromTriggerTime()) if event.isSetUseValuesFromTriggerTime() else True,
        tuple(sorted(assignments)),
    )


def _merge_events(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    species_id_map: Dict[str, str],
    logger: logging.Logger = LOGGER,
) -> None:
    existing = {_event_key(e): e for e in merged_model.getListOfEvents()}
    events2 = list(model2.getListOfEvents())
    events2.sort(key=lambda e: _event_key(e))

    for e2 in events2:
        key = _event_key(e2)
        e1 = existing.get(key)

        if e1 is None:
            clone = e2.clone()
            _rename_species_references(clone, species_id_map)
            rc = merged_model.addEvent(clone)
            if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(f"Failed to add event {key}, libSBML code={rc}")
            existing[key] = clone
            continue

        if _event_signature(e1) != _event_signature(e2):
            logger.warning("Event conflict for key %s. Keeping file1 event.", key)


def merge_remaining_elements(
    merged_model: libsbml.Model,
    model2: libsbml.Model,
    species_id_map: Dict[str, str],
    logger: logging.Logger = LOGGER,
) -> None:
    """Merge remaining SBML entities.

    Includes units, function definitions, rules, and events.
    """

    _merge_unit_definitions(merged_model, model2, logger=logger)
    _merge_function_definitions(merged_model, model2, logger=logger)
    _merge_rules(merged_model, model2, species_id_map=species_id_map, logger=logger)
    _merge_events(merged_model, model2, species_id_map=species_id_map, logger=logger)


def validate_merged_document(
    doc: libsbml.SBMLDocument,
    logger: logging.Logger = LOGGER,
    emit_warnings: bool = True,
) -> None:
    """Validate SBML consistency and fail on errors/fatals."""

    # Run libSBML consistency checks after all merge operations are complete.
    doc.checkConsistency()
    errors = []

    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        sev = err.getSeverity()
        if sev >= libsbml.LIBSBML_SEV_ERROR:
            errors.append(err.getMessage())
        elif sev == libsbml.LIBSBML_SEV_WARNING and emit_warnings:
            logger.warning("Validation warning: %s", err.getMessage())

    if errors:
        preview = " | ".join(errors[:10])
        raise RuntimeError(f"Merged SBML is invalid. Errors: {preview}")


def merge_sbml(
    file1: str,
    file2: str,
    emit_validation_warnings: bool = True,
) -> libsbml.SBMLDocument:
    merged_doc, _ = merge_sbml_with_summary(
        file1,
        file2,
        emit_validation_warnings=emit_validation_warnings,
    )
    return merged_doc


def merge_sbml_with_summary(
    file1: str,
    file2: str,
    emit_validation_warnings: bool = True,
) -> Tuple[libsbml.SBMLDocument, MergeRunSummary]:
    """Merge two SBML files into a deterministic unified SBML document.

    Merge order is deterministic and file1 has priority by default when
    conflicts arise for equivalent entities.
    """

    _configure_default_logging()

    # Read and validate inputs first; merge always starts from a clone of file1.
    doc1 = _read_document(file1)
    doc2 = _read_document(file2)
    model1 = doc1.getModel()

    merged_doc = _clone_document(doc1)
    merged_model = merged_doc.getModel()
    model2 = doc2.getModel()
    assert model1 is not None
    assert merged_model is not None
    assert model2 is not None

    # Track provenance per entity type so we can emit a full end-of-run report.
    compartment_tracker = _EntityTracker(file1_all=_id_list(model1.getListOfCompartments()))
    species_tracker = _EntityTracker(file1_all=_id_list(model1.getListOfSpecies()))
    parameter_tracker = _EntityTracker(file1_all=_id_list(model1.getListOfParameters()))
    reaction_tracker = _EntityTracker(file1_all=_id_list(model1.getListOfReactions()))

    _resolve_model_substance_units(merged_model, model2, logger=LOGGER)

    LOGGER.info("Merging compartments")
    compartment_map = merge_compartments(
        merged_model,
        model2,
        tracker=compartment_tracker,
        logger=LOGGER,
    )

    LOGGER.info("Merging species")
    species_id_map = merge_species(
        merged_model,
        model2,
        compartment_map=compartment_map,
        tracker=species_tracker,
        logger=LOGGER,
    )

    LOGGER.info("Merging parameters")
    merge_parameters(
        merged_model,
        model2,
        policy=MergePolicy(prefer_file1=True),
        tracker=parameter_tracker,
        logger=LOGGER,
    )

    LOGGER.info("Merging reactions")
    merge_reactions(
        merged_model,
        model2,
        species_id_map=species_id_map,
        policy=MergePolicy(prefer_file1=True),
        tracker=reaction_tracker,
        logger=LOGGER,
    )

    LOGGER.info("Merging remaining elements")
    merge_remaining_elements(
        merged_model,
        model2,
        species_id_map=species_id_map,
        logger=LOGGER,
    )

    LOGGER.info("Validating merged model")
    validate_merged_document(
        merged_doc,
        logger=LOGGER,
        emit_warnings=emit_validation_warnings,
    )

    # Freeze trackers into a serializable summary object for reporting.
    summary = MergeRunSummary(
        compartments=compartment_tracker.finalize(),
        species=species_tracker.finalize(),
        parameters=parameter_tracker.finalize(),
        reactions=reaction_tracker.finalize(),
    )

    return merged_doc, summary


def _write_merge_report(summary: MergeRunSummary, output_path: str) -> Path:
    out = Path(output_path)
    # The report is written next to the merged SBML with a deterministic suffix.
    report_path = out.with_suffix(out.suffix + ".merge_report.json")
    report_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return report_path


def _write_document(doc: libsbml.SBMLDocument, output_path: str) -> None:
    rc = libsbml.writeSBMLToFile(doc, output_path)
    # libsbml.writeSBMLToFile returns 1 on success and 0 on failure.
    if rc == 0:
        raise RuntimeError(f"Failed to write output SBML file: {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge two SBML files deterministically")
    parser.add_argument("file1", help="Primary SBML file (priority on conflicts)")
    parser.add_argument("file2", help="Secondary SBML file")
    parser.add_argument(
        "-o",
        "--output",
        default="merged.sbml",
        help="Output SBML file path (default: merged.sbml)",
    )
    parser.add_argument(
        "--suppress-validation-warnings",
        action="store_true",
        help="Do not print SBML consistency warnings during validation",
    )
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Execute merge + provenance collection in a single pass.
    merged_doc, summary = merge_sbml_with_summary(
        args.file1,
        args.file2,
        emit_validation_warnings=not args.suppress_validation_warnings,
    )
    # Persist both the merged SBML and the human/agent-friendly JSON report.
    _write_document(merged_doc, str(out))
    report_path = _write_merge_report(summary, str(out))
    LOGGER.info("Merged SBML written to %s", out)
    LOGGER.info("Merge report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
