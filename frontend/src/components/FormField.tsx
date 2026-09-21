import { useEffect, useState } from "react";
import type { ExtractedFieldOut } from "../api/client";

interface Props {
  field: ExtractedFieldOut;
  threshold: number;
  // Jumps the document viewer to the page this value was read from. The
  // caption used to repeat the source FILENAME under every field, which at
  // 200+ fields is the same string 200+ times. The page it came from is both
  // non-redundant and actionable, so it earns the space instead.
  onJumpToPage?: () => void;
  onSave: (value: string) => Promise<void>;
  onAccept: () => Promise<void>;
  onFocus: () => void;
  isFocused: boolean;
}

/**
 * Put back the space around a contact label the extractor ran together.
 *
 * A letterhead printing "Tel:0422-2441499 Mobile:9629440642" on one line comes
 * back as a single telephone value with the second label welded to the first
 * number — "0422-2441499Mobile:9629440642" — which reads as one impossible
 * 20-digit number.
 *
 * Only the labels are touched, and only where a space is genuinely missing.
 * A blanket "insert a space before every capital" would break the method
 * strings all over these reports: "AOAC 22ndEDITION939.03:2019" and
 * "ALT/SOP/03/01: 2016" are printed that way on the page and must stay as
 * they are, since a reviewer is comparing them against the scan character by
 * character.
 *
 * This is display only. The stored value is untouched and the Extracted JSON
 * panel still shows exactly what the extractor produced — the merge is a real
 * extraction defect, and hiding it in the JSON too would only make it harder
 * to find. Editing the field still saves whatever the reviewer types.
 */
const RUN_ON_LABEL = /([^\s])((?:mobile|mob|telephone|tel|phone|fax|e-?mail)\s*:)\s*/gi;

function spaceRunOnLabels(value: string): string {
  return value.replace(RUN_ON_LABEL, "$1 $2 ");
}

function prettifyLabel(fieldPath: string): string {
  const last = fieldPath.split(".").pop() ?? fieldPath;
  return last
    .replace(/\[\d+\]$/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * One extracted value, as a card: what it is, what was read, where from, how
 * sure the extractor was, and a confirm.
 *
 * The value is an editable input rather than static text. A reviewer who
 * spots "0.8" where the page says "0.3" has to be able to fix it here; the
 * alternative is re-extracting a whole document to correct one digit. It is
 * styled as the 16px bold value it represents and only draws a box when
 * hovered or focused, so a grid of them still reads as values rather than as
 * a wall of form inputs.
 */
export default function FormField({ field, threshold, onJumpToPage, onSave, onAccept, onFocus, isFocused }: Props) {
  const [value, setValue] = useState(() => spaceRunOnLabels(field.value ?? ""));
  const [saving, setSaving] = useState(false);

  // Resync when the server-side value changes under the same field id (an
  // in-place update rather than a fresh row from re-extract) — otherwise the
  // input keeps showing whatever it was seeded with on mount.
  useEffect(() => {
    setValue(spaceRunOnLabels(field.value ?? ""));
  }, [field.id, field.value]);

  const needsReview = field.confidence < threshold;
  const label = prettifyLabel(field.field_path);

  async function commit() {
    // Against the DISPLAYED baseline, not the raw one: the two differ
    // whenever a label was re-spaced, and comparing to the raw value would
    // save that cosmetic change the first time anyone tabbed past the field.
    if (value === spaceRunOnLabels(field.value ?? "")) return;
    setSaving(true);
    try {
      await onSave(value);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className={`field${field.accepted ? " is-confirmed" : ""}${isFocused ? " is-focused" : ""}`}
      onFocus={onFocus}
    >
      <label className="field-label" htmlFor={`field-${field.id}`}>
        {label}
      </label>

      <input
        id={`field-${field.id}`}
        className="field-value"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onFocus={onFocus}
        onBlur={commit}
        disabled={saving}
        placeholder="—"
      />

      <div className="field-foot">
        {field.source_page !== null && (
          <button
            className="field-page"
            onClick={onJumpToPage}
            aria-label={`Show page ${field.source_page + 1}, the source of ${label}`}
          >
            p{field.source_page + 1}
          </button>
        )}
        <span className={`field-conf${needsReview ? " is-low" : ""}`}>
          {Math.round(field.confidence * 100)}%
        </span>
        <button
          className={`field-confirm${field.accepted ? " is-on" : ""}`}
          onClick={onAccept}
          aria-pressed={field.accepted}
          aria-label={field.accepted ? `Unconfirm ${label}` : `Confirm ${label}`}
        >
          {field.accepted ? "Confirmed" : "Confirm"}
        </button>
      </div>
    </div>
  );
}
