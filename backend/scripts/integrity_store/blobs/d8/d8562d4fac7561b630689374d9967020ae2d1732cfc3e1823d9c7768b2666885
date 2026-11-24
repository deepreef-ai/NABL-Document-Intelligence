import { useEffect, useRef, useState } from "react";
import type { ExtractedFieldOut } from "../api/client";

interface Props {
  field: ExtractedFieldOut;
  threshold: number;
  // Jumps the document viewer to the page this value was read from. The
  // caption used to repeat the source FILENAME under every field, which at
  // 200+ fields is the same string 200+ times — and the panel header already
  // names the document. The page it came from is both non-redundant and
  // actionable, so it earns the space instead.
  onJumpToPage?: () => void;
  onSave: (value: string) => Promise<void>;
  onAccept: () => Promise<void>;
  onFocus: () => void;
  isFocused: boolean;
}

function prettifyLabel(fieldPath: string): string {
  const last = fieldPath.split(".").pop() ?? fieldPath;
  return last
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export default function FormField({ field, threshold, onJumpToPage, onSave, onAccept, onFocus, isFocused }: Props) {
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
  const confidenceClass = needsReview ? (field.confidence < 0.5 ? "confidence-low" : "confidence-medium") : "confidence-high";
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
        <span>
          {field.source_page !== null && !isEmpty && (
            <>
              <button className="field-page-link" onClick={onJumpToPage} title={`Go to page ${field.source_page + 1}`}>
                p{field.source_page + 1}
              </button>
              {" · "}
            </>
          )}
          <span className={confidenceClass}>{Math.round(field.confidence * 100)}%</span>
        </span>
        <button className={field.accepted ? "accepted" : ""} onClick={onAccept} title="Confirm this value">
          {field.accepted ? "✓ Confirmed" : "Confirm"}
        </button>
      </div>
    </div>
  );
}
