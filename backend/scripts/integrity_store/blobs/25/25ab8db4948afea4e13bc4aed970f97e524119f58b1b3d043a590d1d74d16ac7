/**
 * Group extracted fields into a two-level form: section, then sub-heading.
 *
 *   Patient                             <- gold section, decided by the backend
 *      Header and Sample Information    <- sub-heading (the document's own)
 *          Patient Name · Sex · D.O.B.
 *   Laboratory
 *      (no heading printed)
 *          Lab Name · Lab Address · Lab Email
 *
 * The section is the labelled dataset's own — `lab_info`, `patient_info`,
 * `signatories` — chosen once on the backend and carried on `field.group`.
 * The sub-heading is the heading the document itself printed. Using both is
 * what makes this read as a form rather than a list: the section says what
 * kind of thing it is, the sub-heading says which part of the page it is from.
 *
 * Nothing here classifies. The frontend used to run its own keyword matcher
 * over field names, which meant the exported JSON and the form on screen could
 * file the same value in two different places.
 *
 * Repeating table rows are a third case — five samples each with an id, a
 * number and a typing. Rendered flat and reflowed across two columns those
 * interleave, so sample 1's id ends up beside sample 2's number. They are
 * detected as a cycle and rebuilt into records.
 *
 * Classification is deterministic keyword matching, never a model: the label
 * sits beside a value someone is about to confirm, so it has to be
 * reproducible rather than merely plausible.
 */

import type { ExtractedFieldOut } from "./client";
import { goldLabel, goldRank } from "./goldSections";

export interface ReviewRow {
  field: ExtractedFieldOut;
}

export interface ReviewRecord {
  label: string;
  rows: ReviewRow[];
}

export interface ReviewSubgroup {
  /** The document's own heading, or "" when it printed none. */
  heading: string;
  rows: ReviewRow[];
  records: ReviewRecord[];
}

export interface ReviewGroup {
  heading: string;
  subgroups: ReviewSubgroup[];
  total: number;
  needsReview: number;
}

const INDEXED = /^([A-Za-z0-9_]+)\[(\d+)\]\.(.+)$/;
const NO_SUBHEADING = "";

function titleCase(s: string): string {
  return s
    .replace(/[_\-.]+/g, " ")
    .trim()
    .split(/\s+/)
    .map((w) => (w.length <= 3 && w === w.toUpperCase() ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}

/**
 * Find a repeating cycle of field names — a results table flattened into a run.
 *
 * A real document is header fields THEN a table, so the cycle rarely starts at
 * index 0; requiring that found no table at all on a report with eight header
 * fields ahead of a five-row table.
 */
function detectCycle(names: string[]): { start: number; len: number } | null {
  const n = names.length;
  if (n < 4) return null;
  for (let start = 0; start <= n - 4; start++) {
    const span = n - start;
    for (let len = 2; len <= Math.floor(span / 2); len++) {
      if (span % len !== 0) continue;
      const head = names.slice(start, start + len);
      // Distinct names is what makes it a record; the same name repeated is a
      // list, and splitting that into rows of one is noise.
      if (new Set(head).size !== len) continue;
      let matches = true;
      for (let block = start + len; block < n && matches; block += len) {
        for (let i = 0; i < len; i++) {
          if (names[block + i] !== head[i]) {
            matches = false;
            break;
          }
        }
      }
      if (matches) return { start, len };
    }
  }
  return null;
}

/** Prefer an identifying value over "Row N" so records are recognisable. */
function recordLabel(rows: ReviewRow[], index: number): string {
  const identifying = rows.find((r) => {
    const leaf = (r.field.field_path.split(".").pop() ?? "").toLowerCase();
    return /(^|_)(id|no|name|number|code)$/.test(leaf) && (r.field.value ?? "").trim();
  });
  if (identifying?.field.value) {
    const leaf = titleCase(identifying.field.field_path.split(".").pop() ?? "");
    return `${leaf}: ${identifying.field.value}`;
  }
  return `Row ${index + 1}`;
}

export function groupReviewFields(
  fields: ExtractedFieldOut[],
  threshold: number,
): ReviewGroup[] {
  // section -> sub-heading -> { rows, records }
  const tree = new Map<string, Map<string, { rows: ReviewRow[]; records: Map<string, ReviewRow[]> }>>();

  const bucket = (section: string, sub: string) => {
    let subs = tree.get(section);
    if (!subs) {
      subs = new Map();
      tree.set(section, subs);
    }
    let b = subs.get(sub);
    if (!b) {
      b = { rows: [], records: new Map() };
      subs.set(sub, b);
    }
    return b;
  };

  const subOf = (f: ExtractedFieldOut) => (f.section ?? "").trim();
  const groupOf = (f: ExtractedFieldOut) => goldLabel(f.group);

  const flat: ExtractedFieldOut[] = [];

  for (const f of fields) {
    const indexed = INDEXED.exec(f.field_path);
    if (indexed) {
      const [, attr, idx] = indexed;
      const b = bucket(groupOf(f), subOf(f) || titleCase(attr));
      const key = `#${idx}`;
      b.records.set(key, [...(b.records.get(key) ?? []), { field: f }]);
      continue;
    }
    if (f.field_path.includes(".")) {
      bucket(groupOf(f), subOf(f)).rows.push({ field: f });
      continue;
    }
    flat.push(f);
  }

  const cycle = detectCycle(flat.map((f) => f.field_path));
  const preamble = cycle ? flat.slice(0, cycle.start) : flat;
  for (const f of preamble) {
    bucket(groupOf(f), subOf(f)).rows.push({ field: f });
  }

  if (cycle) {
    const table = flat.slice(cycle.start);
    const first = table[0];
    const section = groupOf(first);
    const sub = subOf(first) || `${titleCase(first.field_path.replace(/_(id|no|number)$/, ""))} Records`;
    const b = bucket(section, sub);
    for (let start = 0, r = 0; start < table.length; start += cycle.len, r++) {
      b.records.set(`#${r}`, table.slice(start, start + cycle.len).map((field) => ({ field })));
    }
  }

  const groups: ReviewGroup[] = [];
  for (const [heading, subs] of tree) {
    const subgroups: ReviewSubgroup[] = [];
    let total = 0;
    let needsReview = 0;

    for (const [sub, b] of subs) {
      const all = [...b.rows, ...[...b.records.values()].flat()];
      total += all.length;
      needsReview += all.filter((r) => r.field.confidence < threshold).length;
      subgroups.push({
        heading: sub,
        rows: b.rows,
        records: [...b.records.entries()]
          .sort((a, z) => Number(a[0].slice(1)) - Number(z[0].slice(1)))
          .map(([, rows], i) => ({ label: recordLabel(rows, i), rows })),
      });
    }

    // An unnamed sub-heading holds the fields the document did not file
    // anywhere, so it belongs last rather than first.
    subgroups.sort((a, z) => {
      if ((a.heading === NO_SUBHEADING) !== (z.heading === NO_SUBHEADING)) {
        return a.heading === NO_SUBHEADING ? 1 : -1;
      }
      return a.heading.localeCompare(z.heading);
    });

    groups.push({ heading, subgroups, total, needsReview });
  }

  // The order the gold records are written in — who ran the test, for whom, on
  // what, what came back — rather than alphabetical or whichever page happened
  // to come first.
  groups.sort((a, z) => goldRank(a.heading) - goldRank(z.heading));
  return groups;
}
