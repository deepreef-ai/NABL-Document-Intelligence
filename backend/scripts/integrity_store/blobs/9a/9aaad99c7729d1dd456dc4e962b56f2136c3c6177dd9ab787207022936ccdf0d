"""I. Evidence Validation Agent — deterministic, no LLM.

The anti-hallucination gate, and the reason anything downstream is allowed to
call a value "verified".

Every extracted field claims a quote and a page. This node checks the claim
against the actual page text we extracted in preprocessing. It is deliberately
deterministic: asking a model whether a model's output is supported gets you a
second opinion from the same kind of witness, and the whole point here is to
compare against the document rather than against a judgement.

The check runs in two stages, because they can fail independently:

1. **Is the quote real?** The evidence string must occur in the cited page's
   text, after the shared normalisation in helpers. Not found on the cited
   page, but found on another page → WRONG_PAGE, which is a provenance error
   rather than a fabrication. Not found anywhere → UNSUPPORTED.
2. **Does the quote actually contain the value?** A real quote that does not
   contain the value it is offered as evidence for supports nothing. That is
   PARTIALLY_SUPPORTED: the model read the right region and got the value
   wrong, which is a different and more recoverable failure than inventing it.

Only SUPPORTED survives as VERIFIED. Everything else is downgraded, kept, and
reported — never deleted, because a reviewer needs to see what was rejected
and why in order to trust the ones that were not.
"""
from __future__ import annotations

import logging
import re

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, clean_text, err, normalise_for_match, similarity
from app.graph.schemas import (
    EvidenceResult,
    EvidenceStatus,
    ExtractedField,
    ExtractionStatus,
)
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "evidence_validation"

#: A bare token is not evidence — "5" occurs on every page — but length alone
#: is the wrong test for that. "pH: 7.4" is seven characters and is perfectly
#: good evidence: it carries a label AND a value, which is exactly what makes a
#: quote checkable. A pure character threshold rejected it, and short labelled
#: numerics are most of a lab report.
#:
#: So a quote is evidential when it is either long enough to be unambiguous, or
#: structured — at least two alphanumeric runs, i.e. something that reads as
#: "label: value" rather than a lone number.
_MIN_EVIDENCE_CHARS = 8
_MIN_STRUCTURED_TOKENS = 2
_MIN_STRUCTURED_CHARS = 4

_TOKENS = re.compile(r"[a-z0-9]+")


def _is_evidential(evidence: str) -> bool:
    cleaned = clean_text(evidence)
    if not cleaned:
        return False
    if len(cleaned) >= _MIN_EVIDENCE_CHARS:
        return True
    tokens = _TOKENS.findall(cleaned.lower())
    return len(tokens) >= _MIN_STRUCTURED_TOKENS and len(normalise_for_match(cleaned)) >= _MIN_STRUCTURED_CHARS

#: How much of a long quote must match. Full-string equality is too brittle:
#: OCR drops a character, the model tidies a space. The normalisation in
#: helpers already folds the cosmetic differences; this absorbs the rest.
_FUZZY_WINDOW_SLACK = 40


def _find_in_page(needle_norm: str, page_norm: str) -> bool:
    return bool(needle_norm) and needle_norm in page_norm


def _fuzzy_in_page(needle_norm: str, page_norm: str, threshold: float) -> float:
    """Best similarity of `needle` against any same-length window of the page.

    Scanning every offset would be O(n*m) on a long page for no real gain, so
    the window slides in steps proportional to the needle. Good enough to tell
    "this quote is roughly here" from "this quote is not in this page".
    """
    n = len(needle_norm)
    if not n or len(page_norm) < n // 2:
        return 0.0
    width = n + _FUZZY_WINDOW_SLACK
    step = max(1, n // 4)
    best = 0.0
    for start in range(0, max(1, len(page_norm) - width + 1), step):
        window = page_norm[start:start + width]
        score = similarity(needle_norm, window)
        if score > best:
            best = score
            if best >= threshold:
                break
    return best


#: A quote may be broken by text the model could not have meant to include.
#: MEASURED on a two-column cytology report: the right-hand sidebar sits flush
#: against the results table, so reading order emits
#:
#:     Qualitative  Abnormal  Normal  Reference Range Epithelial Fragments:
#:     pH  6.5  5-9  Urothelial Fragment
#:
#: The label and its value ARE both on the page, one directly above the other
#: in the sidebar, and the model quoted them as the pair a reader sees. But
#: "pH 6.5 5-9" from the other column falls between them, so the quote is not a
#: contiguous substring and nine correct values on one document were thrown out
#: as unsupported.
#:
#: So: the quote's words must still all appear, in order, on the cited page —
#: that part is not relaxed — but a small amount of page text is allowed to sit
#: between them. Small is the whole safeguard. A column cell is a few words; a
#: quote assembled from scattered fragments of a page is not, and still fails.
#:
#: Three words is the floor rather than four because "Concretions: / Calcified
#: (Lithiasis)" is a real sidebar pair and only three; the length floor is what
#: stops three SHORT words from matching by coincidence.
_MAX_INTERPOSED_CHARS = 48
_MAX_INTERPOSED_RUNS = 3
_MIN_GAPPED_TOKENS = 3
_MIN_GAPPED_CHARS = 16

_WORD = re.compile(r"[0-9a-z]+")


def _tokenise(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _gapped_match(quote_tokens: list[str], page_tokens: list[str]) -> bool:
    """Do the quote's words occur on the page in order, barely interrupted?

    Gap size is counted in CHARACTERS OF INTERVENING WORDS, not in raw span:
    a column layout puts a lot of whitespace between cells and none of that is
    interposed content. Every start position is tried and the smallest total
    gap wins, because the first occurrence of the opening word is often not the
    one the model quoted.
    """
    if len(quote_tokens) < _MIN_GAPPED_TOKENS:
        return False
    if sum(len(t) for t in quote_tokens) < _MIN_GAPPED_CHARS:
        return False

    first = quote_tokens[0]
    for start in (i for i, t in enumerate(page_tokens) if t == first):
        pos, gap_chars, runs, ok = start + 1, 0, 0, True
        for want in quote_tokens[1:]:
            step = 0
            while pos + step < len(page_tokens) and page_tokens[pos + step] != want:
                gap_chars += len(page_tokens[pos + step])
                step += 1
            if pos + step >= len(page_tokens) or gap_chars > _MAX_INTERPOSED_CHARS:
                ok = False
                break
            if step:
                runs += 1
                if runs > _MAX_INTERPOSED_RUNS:
                    ok = False
                    break
            pos += step + 1
        if ok:
            return True
    return False


def validate_evidence(state: GraphState) -> dict:
    settings = get_graph_settings()
    threshold = settings.evidence_similarity_threshold

    fields = [f.model_copy(deep=True) for f in state.active_fields()]
    page_norm = {p: normalise_for_match(t) for p, t in state.page_text.items()}

    results: list[EvidenceResult] = []
    entries = []
    errors = []
    counts = {s: 0 for s in EvidenceStatus}

    for f in fields:
        uid = f.field_uid
        evidence = f.exact_source_evidence or ""
        evidence_norm = normalise_for_match(evidence)
        cited = f.page_number

        # --- no usable quote ----------------------------------------------
        if not _is_evidential(evidence) or not evidence_norm:
            status = EvidenceStatus.NO_EVIDENCE
            results.append(EvidenceResult(
                field_uid=uid, status=status, cited_page=cited,
                reason="no evidence quote, or too short to be evidential",
            ))
            f.evidence_status = status
            if f.extraction_status == ExtractionStatus.VERIFIED:
                f.extraction_status = ExtractionStatus.UNCERTAIN
            counts[status] += 1
            errors += err(
                ErrorType.MISSING_EVIDENCE,
                f"field {f.normalized_field_name!r} has no usable evidence quote",
                NODE, page_number=cited, chunk_id=f.chunk_id,
                recovery_action="downgraded to UNCERTAIN",
                resolution_status="RECOVERED",
            )
            continue

        # --- is the quote on the cited page? ------------------------------
        cited_text = page_norm.get(cited or -1, "")
        found_page: int | None = None
        score = 0.0

        quote_tokens = _tokenise(evidence)
        page_tokens = {p: _tokenise(t) for p, t in state.page_text.items()}
        interposed = False

        if _find_in_page(evidence_norm, cited_text):
            found_page, score = cited, 1.0
        else:
            score = _fuzzy_in_page(evidence_norm, cited_text, threshold)
            if score >= threshold:
                found_page = cited
            elif _gapped_match(quote_tokens, page_tokens.get(cited or -1, [])):
                # Every word is here, in order, with only a column cell between
                # them. The page supports the value; the reading order simply
                # could not put the pair on one line.
                found_page, score, interposed = cited, 1.0, True
            else:
                # Try every other page before calling it a fabrication. A
                # wrong page number is a citation bug; declaring it invented
                # would discard a value that is genuinely in the document.
                for page, text in page_norm.items():
                    if page == cited:
                        continue
                    if _find_in_page(evidence_norm, text):
                        found_page, score = page, 1.0
                        break
                if found_page is None:
                    for page, text in page_norm.items():
                        if page == cited:
                            continue
                        s = _fuzzy_in_page(evidence_norm, text, threshold)
                        if s >= threshold:
                            found_page, score = page, s
                            break
                if found_page is None:
                    for page, tokens in page_tokens.items():
                        if page == cited:
                            continue
                        if _gapped_match(quote_tokens, tokens):
                            found_page, score, interposed = page, 1.0, True
                            break

        if found_page is None:
            status = EvidenceStatus.UNSUPPORTED
            reason = "evidence quote does not occur anywhere in the document"
            errors += err(
                ErrorType.HALLUCINATED_VALUE,
                f"field {f.normalized_field_name!r}: {reason}",
                NODE, page_number=cited, chunk_id=f.chunk_id,
                recovery_action="value rejected, kept for review as UNSUPPORTED",
                resolution_status="RECOVERED",
            )
        elif found_page != cited:
            status = EvidenceStatus.WRONG_PAGE
            reason = f"evidence found on page {found_page}, not the cited page {cited}"
            errors += err(
                ErrorType.INCORRECT_PAGE_REFERENCE,
                f"field {f.normalized_field_name!r}: {reason}",
                NODE, page_number=cited, chunk_id=f.chunk_id,
                recovery_action=f"page corrected to {found_page}",
                resolution_status="RECOVERED",
            )
            # Correcting the page is safe and preserves the value: we know
            # exactly where the quote really is.
            f.page_number = found_page
            f.notes = (f.notes + f" | page corrected from {cited} to {found_page}").strip(" |")
        else:
            # --- quote is real and on the right page. Does it hold the value?
            value_norm = normalise_for_match(f.value)
            if f.value is None:
                # A NOT_FOUND row with a real quote is a legitimate, useful
                # record: "this label exists here and is blank".
                status = EvidenceStatus.SUPPORTED
                reason = "null value, evidence confirms the label is present"
            elif not value_norm:
                status = EvidenceStatus.PARTIALLY_SUPPORTED
                reason = "value normalises to nothing"
            elif value_norm in normalise_for_match(evidence):
                status = EvidenceStatus.SUPPORTED
                reason = "value occurs verbatim inside the quoted evidence"
            elif similarity(f.value, evidence) >= threshold:
                status = EvidenceStatus.SUPPORTED
                reason = "value matches the quoted evidence closely"
            elif value_norm in cited_text:
                # Value is on the page, just not inside the quote the model
                # chose. The value is real; the citation is imprecise.
                status = EvidenceStatus.PARTIALLY_SUPPORTED
                reason = "value is on the cited page but not inside the quoted evidence"
            else:
                status = EvidenceStatus.PARTIALLY_SUPPORTED
                reason = "quote is genuine but does not contain the extracted value"

        # Say so in the audit trail. A quote that only matched across a column
        # break is genuine but was assembled by the reader's eye rather than
        # copied off one line, and anyone auditing a value deserves to know
        # which of the two it was.
        if interposed:
            reason += " (matched across a column break)"

        results.append(EvidenceResult(
            field_uid=uid, status=status, cited_page=cited, found_on_page=found_page,
            matched_text=evidence[:300], similarity=round(score, 3), reason=reason,
        ))
        f.evidence_status = status
        counts[status] += 1

        # --- status reconciliation -----------------------------------------
        if status == EvidenceStatus.SUPPORTED:
            if (
                f.extraction_status == ExtractionStatus.UNCERTAIN
                and f.confidence_score >= settings.verified_confidence_threshold
            ):
                # Evidence is the stronger signal: a value we can see in the
                # document is verified regardless of how the model felt.
                f.extraction_status = ExtractionStatus.VERIFIED
        elif status in (EvidenceStatus.PARTIALLY_SUPPORTED, EvidenceStatus.WRONG_PAGE):
            if f.extraction_status == ExtractionStatus.VERIFIED:
                f.extraction_status = ExtractionStatus.UNCERTAIN
        else:  # UNSUPPORTED / NO_EVIDENCE
            f.extraction_status = ExtractionStatus.UNCERTAIN
            f.notes = (f.notes + f" | {reason}").strip(" |")

    summary = ", ".join(f"{s.value}={counts[s]}" for s in EvidenceStatus if counts[s])
    entries += audit(
        NODE, "evidence_validated", f"{len(fields)} field(s): {summary or 'none'}",
        level="warning" if counts[EvidenceStatus.UNSUPPORTED] else "info",
    )

    return {
        "current_node": NODE,
        "normalized_fields": fields,
        "evidence_validation_results": results,
        "audit_log": entries,
        "errors": errors,
    }
