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
 *
 * ORDER follows the document, not the taxonomy. A reviewer reads the page on
 * the left and the form on the right at the same time, so the two have to run
 * in the same direction: a report that prints the patient block at the top of
 * page 1 gets a form that opens on Patient. Three signals decide, in order of
 * how much they can be trusted:
 *
 * 1. **Where the field was found** — page, then how far down the page. Only
 *    the grounded pipeline matches values back to a rectangle, so this is
 *    exact when it exists and absent when it does not.
 * 2. **The order the fields arrived in.** Chunked extraction reads the pages
 *    in order and returns fields in the order it read them, which on a real
 *    report runs patient, sample dates, results, footer, signature. This is
 *    what actually orders the form today: nothing in the current pipeline
 *    carries a bbox. It is a weaker signal than a rectangle — schema-driven
 *    rule extraction returns fields in the order of the SCHEMA — which is why
 *    a rectangle outranks it rather than the other way round.
 * 3. **The gold reading order**, for whatever the first two cannot separate.
 *
 * A section sits where it STARTS — the earliest of its fields — because that
 * is what "the patient block is at the top" means to someone reading the page.
 * The cost is that one misfiled field pulls its whole section up with it. The
 * alternative, a median, is steadier under misclassification but files a
 * section that opens at the top of page 1 below one that opens halfway down,
 * which is the worse failure: it breaks the one property this ordering exists
 * to provide, that the form and the page can be read side by side.
 */

import type { ExtractedFieldOut } from "./client";
import { goldLabel, goldRank } from "./goldSections";

export interface ReviewRow {
  field: ExtractedFieldOut;
}

/**
 * Where a set of fields sits in the document.
 *
 * `rect` is the page the earliest field was read from and how far down that
 * page it sat, or null when not even a page is known. `y` is Infinity for a
 * field whose page is known but which was never matched to a rectangle —
 * page 1 is still ahead of page 2, and within its own page it sorts after
 * everything that can say where it was. `arrival` is the earliest position in
 * the order the fields came back from extraction, which every field has.
 */
export interface DocPlace {
  rect: { page: number; y: number } | null;
  arrival: number;
}

/** The front of an already-sorted list; undefined when there is nothing. */
function first<T>(sorted: T[]): T | undefined {
  return sorted[0];
}

/** Where these fields sit: where the earliest of them was found. */
function placeOf(fields: ExtractedFieldOut[], arrivalOf: (f: ExtractedFieldOut) => number): DocPlace {
  const located = fields
    .filter((f) => f.source_page != null)
    .map((f) => ({ page: f.source_page as number, y: f.source_bbox?.y ?? Number.POSITIVE_INFINITY }))
    .sort((a, z) => a.page - z.page || (a.y === z.y ? 0 : a.y - z.y));

  return {
    rect: first(located) ?? null,
    // An empty section sorts last rather than first — it says nothing about
    // where it belongs, and guessing "the top" is the costly way to be wrong.
    arrival: first(fields.map(arrivalOf).sort((a, z) => a - z)) ?? Number.POSITIVE_INFINITY,
  };
}

/**
 * Sort comparator over document places: rectangles first where both sides
 * have one, then the order extraction returned the fields in. 0 means the two
 * sit in the same spot and the caller falls back to its own order.
 *
 * A section with rectangles is NOT ranked above one without. The two scales
 * are unrelated — points down a page against a position in a list — so a
 * mixed pair is compared on the signal they both have.
 */
export function comparePlaces(a: DocPlace, z: DocPlace): number {
  if (a.rect && z.rect) {
    // Equality before subtraction: two ungrounded pages are both Infinity,
    // and Infinity - Infinity is NaN, which a comparator must never return.
    const byRect = a.rect.page - z.rect.page || (a.rect.y === z.rect.y ? 0 : a.rect.y - z.rect.y);
    if (byRect !== 0) return byRect;
  }
  // Equality first: two empty sections are both Infinity, and subtracting
  // those gives NaN, which a sort comparator must never see.
  return a.arrival === z.arrival ? 0 : a.arrival - z.arrival;
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
  place: DocPlace;
}

export interface ReviewGroup {
  heading: string;
  subgroups: ReviewSubgroup[];
  total: number;
  needsReview: number;
  place: DocPlace;
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
  // The order extraction returned the fields in — see the ordering note at the
  // top of the file. Captured before anything is grouped, because grouping is
  // what loses it.
  const arrival = new Map(fields.map((f, i) => [f.id, i]));
  const arrivalOf = (f: ExtractedFieldOut) => arrival.get(f.id) ?? fields.length;
  const place = (rows: ReviewRow[]) => placeOf(rows.map((r) => r.field), arrivalOf);

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

    const groupRows: ReviewRow[] = [];

    for (const [sub, b] of subs) {
      const all = [...b.rows, ...[...b.records.values()].flat()];
      total += all.length;
      needsReview += all.filter((r) => r.field.confidence < threshold).length;
      groupRows.push(...all);
      subgroups.push({
        heading: sub,
        rows: b.rows,
        records: [...b.records.entries()]
          .sort((a, z) => Number(a[0].slice(1)) - Number(z[0].slice(1)))
          .map(([, rows], i) => ({ label: recordLabel(rows, i), rows })),
        place: place(all),
      });
    }

    // Sub-headings run in the order the document printed them. Alphabetical
    // stays the fallback: it is at least stable, and an unnamed sub-heading —
    // the fields the document did not file anywhere — belongs last.
    subgroups.sort((a, z) => {
      const byPlace = comparePlaces(a.place, z.place);
      if (byPlace !== 0) return byPlace;
      if ((a.heading === NO_SUBHEADING) !== (z.heading === NO_SUBHEADING)) {
        return a.heading === NO_SUBHEADING ? 1 : -1;
      }
      return a.heading.localeCompare(z.heading);
    });

    groups.push({ heading, subgroups, total, needsReview, place: place(groupRows) });
  }

  // Document order, so the form runs down the page the way the page does: the
  // patient block above the results if that is how the report was printed.
  // Gold reading order — who ran the test, for whom, on what, what came back —
  // settles whatever the document cannot.
  groups.sort((a, z) => comparePlaces(a.place, z.place) || goldRank(a.heading) - goldRank(z.heading));
  return groups;
}
