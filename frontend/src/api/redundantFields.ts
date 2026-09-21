import type { ExtractedFieldOut, TestRowOut } from "./client";

/**
 * Drop scalar fields that only repeat a cell of the results table.
 *
 * A lab report prints each analyte's method under the analyte, inside the
 * results table. Extraction picks that up twice: once as the table's `method`
 * column, and once as a loose key-value pair, because "METHOD : BIURATE" on
 * its own line looks exactly like every other label-value pair on the page.
 * The review form then shows seven cards reading BIURATE, BROMOCRESOLGREEN,
 * LACTATE-PYRUVATE — the same seven strings already sitting in the Method
 * column a few inches above, each with its own Confirm button and each
 * counting against the progress bar.
 *
 * The rule is deliberately narrow, because hiding an extracted value is not a
 * free action — a hidden field cannot be confirmed or corrected. A field is
 * folded only when BOTH hold:
 *
 *   1. its name says it is a method — `method_biurate`, `test_protocol`;
 *   2. its value is already in a method-ish column of this document's own
 *      results table.
 *
 * So a method the table does NOT carry survives and is still shown. Matching
 * on the value rather than on the name alone is the whole point: "drop
 * anything called method" would silently lose the one document whose method
 * was printed in a header rather than in the table.
 *
 * Nothing is deleted. The fields stay in the API response and in the
 * Extracted JSON panel; this only decides what the form asks a person to
 * confirm.
 */

const METHOD_NAME = /(^|[._])(method|protocol|technique)([._]|$)/i;
const METHOD_COLUMN = /(method|protocol|technique)/i;

/**
 * Compare on what the words say, not how they were spaced.
 *
 * The same method reaches the two places in different shapes: the table has
 * "DIAZO METHOD" where the loose field has "METHOD:DIAZO", and OCR drops the
 * spaces from "COLORIMETRIC DIAZO METHOD" in one and not the other. Case,
 * punctuation, spacing and the word "method" itself all carry no meaning
 * here, so all four are removed from both sides before comparing.
 */
function normalise(value: string): string {
  return value
    .toLowerCase()
    .replace(/method/g, "")
    .replace(/[^a-z0-9]/g, "");
}

/** Every method-ish value the results table already shows. */
function methodsInTable(tests: TestRowOut[]): Set<string> {
  const seen = new Set<string>();
  for (const row of tests) {
    for (const [column, value] of Object.entries(row)) {
      if (!METHOD_COLUMN.test(column)) continue;
      if (value === null || value === undefined) continue;
      const key = normalise(String(value));
      if (key) seen.add(key);
    }
  }
  return seen;
}

export interface ReviewFields {
  /** What the form shows and counts. */
  fields: ExtractedFieldOut[];
  /** Fields folded away because the table already carries them. */
  folded: ExtractedFieldOut[];
}

export function splitRedundantFields(
  fields: ExtractedFieldOut[],
  tests: TestRowOut[],
): ReviewFields {
  if (tests.length === 0) return { fields, folded: [] };

  const inTable = methodsInTable(tests);
  if (inTable.size === 0) return { fields, folded: [] };

  const kept: ExtractedFieldOut[] = [];
  const folded: ExtractedFieldOut[] = [];

  for (const field of fields) {
    const leaf = field.field_path.split(".").pop() ?? field.field_path;
    const value = (field.value ?? "").trim();
    const duplicate =
      value !== "" && METHOD_NAME.test(leaf) && inTable.has(normalise(value));
    (duplicate ? folded : kept).push(field);
  }

  return { fields: kept, folded };
}
