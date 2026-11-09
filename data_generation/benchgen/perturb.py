"""S9: paraphrased and perturbed answers for Truth Ratio, attached to T1/T2 evaluation rows."""
from __future__ import annotations

from .config import Config
from .prompts import perturbations_prompt
from .schemas import PerturbationsOut
from .textutil import contains_entity, content_words, normalize, word_count

N_PERTURBED = 5


def _sim(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def key_entities(fact: dict, entity: dict) -> list[str]:
    if fact["role"] == "incident":
        return list(entity["E_f"])
    keys = list(fact["own_entities"]) + list(fact["measured"]["shared_entities"])
    if fact["measured"]["anchor_present"]:
        keys.append(entity["A_f"])
    return keys


MIN_PARAPHRASE_SIM = 0.75  # a one-detail edit keeps most of the paraphrase's character sequence


def validate_perturbation(text: str, keys: list[str], paraphrase: str, perturbed: list[str]) -> list[str]:
    """Perturbed answers are variants of the PARAPHRASE, not of the original sentence.

    Truth Ratio compares P(perturbed) against P(paraphrase). In the pilot the perturbations kept
    the gold sentence's wording while the paraphrase reworded it, so a model that had memorized
    the gold wording scored the *false* answers higher (FT's incident Truth Ratio was 4.57, and a
    single question reached 19.7). Surface form, not truth, was being measured. Requiring each
    variant to differ from the paraphrase by one detail makes the pair surface-matched, which is
    also what makes the derived knowledge probe meaningful.
    """
    reasons = []
    n = word_count(text)
    if normalize(paraphrase) == normalize(text):
        reasons.append("paraphrase must reword the text")
    missing = [k for k in keys if not contains_entity(paraphrase, k)]
    if missing:
        reasons.append(f"paraphrase dropped named entities: {missing}")
    if not 0.5 * n <= word_count(paraphrase) <= 1.8 * n:
        reasons.append("paraphrase length too different from the original")
    if len(perturbed) != N_PERTURBED:
        reasons.append(f"need exactly {N_PERTURBED} perturbed answers, got {len(perturbed)}")
    seen = {normalize(text), normalize(paraphrase)}
    para_words, n_para = content_words(paraphrase), word_count(paraphrase)
    for p in perturbed:
        if normalize(p) in seen:
            reasons.append("perturbed answers must differ from the original, the paraphrase, and each other")
            break
        seen.add(normalize(p))
        if not content_words(p) ^ para_words:
            reasons.append(f"perturbed answer does not change any entity or detail of the paraphrase: {p!r}")
        # Which sentence is it a variant of? Counting changed words cannot tell, because a
        # paraphrase reuses most of the original's content words. Sequence similarity can.
        elif _sim(p, text) > _sim(p, paraphrase):
            reasons.append(f"perturbed answer is a variant of the original sentence; it must be a variant "
                           f"of the paraphrase: {p!r}")
        elif _sim(p, paraphrase) < MIN_PARAPHRASE_SIM:
            reasons.append(f"perturbed answer must be the paraphrase with ONE detail changed, not a rewording: {p!r}")
        if not 0.7 * n_para <= word_count(p) <= 1.4 * n_para:
            reasons.append("perturbed answer length must match the paraphrase")
    return reasons


def generate_perturbations(llm, cfg: Config, entity: dict, facts: list[dict],
                           eval_rows: list[dict]) -> dict[str, list[str]]:
    """Fills `paraphrase` and `perturbed` in place on fact_text eval rows. Returns failures."""
    by_id = {f["fact_id"]: f for f in facts}
    pending = {fid: {"fact_id": fid, "text": f["text"]} for fid, f in by_id.items()}
    accepted, failures = {}, {}
    for rnd in range(cfg.repair_rounds + 1):
        note = ""
        if failures:
            note = "\n\nFix these rejected facts:\n" + "\n".join(f"- {fid}: {'; '.join(r)}" for fid, r in failures.items())
        out = llm.call("perturbations", perturbations_prompt(list(pending.values())) + note, PerturbationsOut,
                       tag=f"{entity['entity_id']}/perturbations/r{rnd}")
        if out is None:
            continue
        got = {p.fact_id: p for p in out.facts}
        failures = {}
        for fid in list(pending):
            p = got.get(fid)
            if p is None:
                failures[fid] = ["fact missing from the response"]
                continue
            fact = by_id[fid]
            reasons = validate_perturbation(fact["text"], key_entities(fact, entity), p.paraphrase.strip(),
                                            [x.strip() for x in p.perturbed])
            if reasons:
                failures[fid] = reasons
            else:
                accepted[fid] = (p.paraphrase.strip(), [x.strip() for x in p.perturbed])
                pending.pop(fid)
        llm.log.record_validation(task="perturbations", tag=f"{entity['entity_id']}/perturbations/r{rnd}",
                                  failures={fid: failures[fid] for fid in pending if fid in failures})
        if not pending:
            break

    for row in eval_rows:
        if row["gold_form"] == "fact_text" and row["fact_id"] in accepted:
            row["paraphrase"], row["perturbed"] = accepted[row["fact_id"]]
    return {fid: failures.get(fid, ["no response"]) for fid in pending}
