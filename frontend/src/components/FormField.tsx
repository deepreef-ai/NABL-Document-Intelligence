import { useEffect, useRef, useState } from "react";
import type { ExtractedFieldOut } from "../api/client";

interface Props {
  field: ExtractedFieldOut;
  threshold: number;
  // Jumps the document viewer to the page this value was read from.
  //
  // Nothing calls it at the moment: the "p1" chip in the caption was its
  // only trigger and that has been removed. Kept on the interface so
  // ReviewFieldGroups still compiles and so the behaviour can be re-hung on
  // something else — the label, or the field itself on focus — without
  // rewiring the chain again.
  onJumpToPage?: () => void;
  onSave: (value: string) => Promise<void>;
  onAccept: () => Promise<void>;
  onFocus: () => void;
  isFocused: boolean;
}

/**
 * Initialisms that must not be title-cased.
 *
 * Every one of these is a real leaf name in this project's own extractions —
 * "Cin", "Gst", "Ulr" and "Cmc" are not words, and a reviewer scanning a
 * column of labels reads them as typos before they read them as the
 * identifiers they are.
 */
const INITIALISMS = new Set([
  "cin", "gst", "gstin", "pan", "tan", "ulr", "cmc", "id", "nabl", "iso",
  "iec", "pt", "ilc", "sop", "cv", "uv", "ifcc", "hsn", "msme", "qr", "url",
  "rm", "sl", "sr",
]);

/**
 * Words that stay lowercase inside a label, never at the front.
 *
 * Title-casing every token turns "date_of_receipt" into "Date Of Receipt"
 * and "range_and_accuracy" into "Range And Accuracy", which reads like a
 * headline rather than a form label.
 */
const MINOR_WORDS = new Set([
  "of", "and", "or", "by", "on", "in", "to", "for", "the", "a", "an",
  "with", "at", "per", "from", "as", "vs",
]);

/**
 * A field's leaf name, as a human label.
 *
 * Only the leaf: "senior_management.head_of_laboratory.name" is shown under
 * a heading that already says "Senior Management", so repeating the path
 * would be the same words twice.
 */
function prettifyLabel(fieldPath: string): string {
  const last = fieldPath.split(".").pop() ?? fieldPath;
  return last
    .split("_")
    .filter(Boolean)
    .map((word, i) => {
      const lower = word.toLowerCase();
      if (INITIALISMS.has(lower)) return lower.toUpperCase();
      // A bare number is a unit or a step ("phosphate_5"), not a word.
      if (/^\d+$/.test(word)) return word;
      if (i > 0 && MINOR_WORDS.has(lower)) return lower;
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

export default function FormField({ field, threshold, onSave, onAccept, onFocus, isFocused }: Props) {
  const [value, setValue] = useState(field.value ?? "");
  const [saving, setSaving] = useState(false);
  const [justFilled, setJustFilled] = useState(false);
  const previousServerValue = useRef(field.value);

  // Resync when the server-side value changes under the same field id (e.g.
  // an in-place update rather than a fresh row from re-extract) — otherwise
  // the input keeps showing whatever it was seeded with on mount.
  useEffect(() => {
    setValue(field.value ?? "");
  }, [field.id, field.value]);

  // A brief highlight flash the moment a blank field gets a real extracted
  // value — the "before/after" cue that something just got auto-filled,
  // rather than every field looking identically inert.
  useEffect(() => {
    if (field.value && field.value !== previousServerValue.current) {
      setJustFilled(true);
      const t = setTimeout(() => setJustFilled(false), 1400);
      previousServerValue.current = field.value;
      return () => clearTimeout(t);
    }
    previousServerValue.current = field.value;
  }, [field.value]);

  const needsReview = field.confidence < threshold;
  const isEmpty = !value.trim();

  async function commit() {
    if (value === field.value) return;
    setSaving(true);
    try {
      await onSave(value);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={`form-field ${isFocused ? "form-field-focused" : ""}`} onFocus={onFocus}>
      <label>
        {prettifyLabel(field.field_path)}
        {field.accepted ? (
          <svg className="field-status-icon field-status-confirmed" viewBox="0 0 20 20" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2">
            <path d="M4 10.5l4 4 8-9" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : needsReview ? (
          <svg className="field-status-icon field-status-warning" viewBox="0 0 20 20" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10 3 2 17h16L10 3z" strokeLinejoin="round" />
            <path d="M10 8.5v3.5" strokeLinecap="round" />
            <circle cx="10" cy="14.2" r="0.6" fill="currentColor" stroke="none" />
          </svg>
        ) : null}
      </label>

      <input
        className={`form-field-value ${isEmpty ? "form-field-value-empty" : ""} ${justFilled ? "form-field-value-flash" : ""}`}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onFocus={onFocus}
        onBlur={commit}
        disabled={saving}
        placeholder="—"
      />

      <div className="form-field-caption">
        {/* No "p1 · 100%" here any more.
            The page chip said p1 on every row — source_page is 0 for every
            field of a single-page document, which is most of them — so it
            carried no information while taking the eye first. The confidence
            percentage was read as a score for the value, which it is not; a
            value the model is unsure of is already called out by the
            needs-review styling on the input above, where it belongs. */}
        <span />
        {/* A tick, not the words Confirm / Confirmed. There is one of these
            on every row, so the label repeated down the whole column and read
            as noise; the state is binary and a tick shows it at a glance.
            Fixed size, so the control does not change width between its two
            states and nudge every row as you work down the list. */}
        <button
          className={`field-tick${field.accepted ? " accepted" : ""}`}
          onClick={onAccept}
          aria-pressed={field.accepted}
          aria-label={field.accepted ? "Confirmed — click to undo" : "Confirm this value"}
          title={field.accepted ? "Confirmed — click to undo" : "Confirm this value"}
        >
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
               strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="m4.5 12.5 4.5 4.5L19.5 6.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}
