import type { TestRowOut } from "../api/client";

/**
 * The results table, as a table.
 *
 * A results table is not a list of fields, and treating it as one is the
 * single biggest way an extraction can be technically complete and
 * practically useless. Flattened into scalars, a urine panel's `pH 6.5 5-9`
 * becomes one string with the reading and the reference range fused together:
 * nothing to sort by, nothing to compare, no way to see at a glance that a
 * result sits outside its range. The labelled dataset keeps the columns
 * separate, and so does this.
 *
 * Columns are chosen from the data rather than fixed, because the 53 gold
 * records do not agree on which exist — `reference_range` in one lab's
 * format, `bio_ref_interval` in another's, `test_protocol` in a third's. A
 * fixed set would show four empty columns on most documents and silently drop
 * whatever the fifth lab happened to print.
 *
 * Rows group under their panel heading ("URINE CHEMISTRY"), which is how the
 * page printed them and how a reviewer checking against the PDF will read.
 */

interface Props {
  tests: TestRowOut[];
}

/** Column order where a column exists; anything unrecognised follows, sorted. */
const PREFERRED = [
  "s_no",
  "test_name",
  "test_parameter",
  "parameter",
  "analyte",
  "result",
  "unit",
  "reference_range",
  "bio_ref_interval",
  "normal_range",
  "limit",
  "flag",
  "method",
  "test_method",
  "test_protocol",
  "specimen",
  "sample_date",
];

/** Never a column: grouping keys, provenance, and the evidence quote. */
const HIDDEN = new Set(["panel_name", "section", "page_number", "exact_source_evidence"]);

const NUMERICISH = /\d/;

function label(column: string): string {
  return column
    .split("_")
    .filter(Boolean)
    .map((w) => (w.length <= 2 ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}

function used(tests: TestRowOut[]): string[] {
  const present = new Set<string>();
  for (const row of tests) {
    for (const [k, v] of Object.entries(row)) {
      if (HIDDEN.has(k)) continue;
      if (v === null || v === undefined || String(v).trim() === "") continue;
      present.add(k);
    }
  }
  const known = PREFERRED.filter((c) => present.has(c));
  const rest = [...present].filter((c) => !PREFERRED.includes(c)).sort();
  return [...known, ...rest];
}

/**
 * Which columns get tabular figures.
 *
 * Decided from the data rather than a fixed list: lining up "12.4" over
 * "9.81" is what makes a column of results scannable, and applying the same
 * to a column of test names would only widen the spaces between letters. A
 * column counts as numeric when most of its filled cells contain a digit.
 */
function numericColumns(tests: TestRowOut[], columns: string[]): Set<string> {
  const numeric = new Set<string>();
  for (const c of columns) {
    const filled = tests
      .map((r) => (r[c] === null || r[c] === undefined ? "" : String(r[c]).trim()))
      .filter(Boolean);
    if (filled.length === 0) continue;
    if (filled.filter((v) => NUMERICISH.test(v)).length * 2 > filled.length) numeric.add(c);
  }
  return numeric;
}

/**
 * What the document said about this reading, normalised for display.
 *
 * Read from the word the document printed, never from comparing the result to
 * the range ourselves — that would be us making a clinical claim the document
 * did not make. "H", "HI" and "High" all mean the same thing to the lab that
 * printed them, so they are shown the same way; anything unrecognised keeps
 * its own text.
 */
function readFlag(value: string): { tone: "high" | "low" | "normal" | "other"; text: string } | null {
  const v = value.trim();
  if (!v) return null;
  const k = v.toLowerCase();
  if (k === "+" || /^(h|hi|high|abnormal|abn|critical|panic)\b/.test(k)) return { tone: "high", text: "High" };
  if (k === "-" || /^(l|lo|low)\b/.test(k)) return { tone: "low", text: "Low" };
  if (/^(n|normal|wnl|ok)\b/.test(k)) return { tone: "normal", text: "Normal" };
  return { tone: "other", text: v };
}

/** "URINE CHEMISTRY" + "Qualitative" -> one heading, skipping either if absent. */
function panelOf(row: TestRowOut): string {
  return [row.panel_name, row.section].filter((p) => p && String(p).trim()).join(" · ");
}

/** The serial number as a number, or null where the document printed none. */
function serial(row: TestRowOut): number | null {
  const raw = row.s_no;
  if (raw === null || raw === undefined) return null;
  const n = Number(String(raw).trim().replace(/[.)]$/, ""));
  return Number.isFinite(n) ? n : null;
}

/**
 * Cut the rows into the tables the document actually printed.
 *
 * A panel heading alone is not enough. A nutrition report prints two separate
 * tables — the proximates, then the fats and minerals — under one heading,
 * "RESULTS OF THE ANALYSIS", and extraction tags every row with that same
 * heading. Grouped by heading, the two arrive as one fifteen-row table whose
 * serial column reads 1-9 and then 1-6 again, which is the clearest possible
 * sign that it is really two tables and the least readable way to show them.
 *
 * So a new table starts when the heading changes OR when the serial number
 * stops increasing. The restart is the document's own boundary marker; only
 * a pair of parsed numbers is trusted, so a table with no serial column, or
 * one whose serials are "1a"/"1b", simply stays whole rather than being cut
 * in arbitrary places.
 *
 * Segments are contiguous, never merged by name across a gap: the order rows
 * were printed in is the order a reviewer reads them against the page.
 */
function segment(tests: TestRowOut[]): { name: string; rows: TestRowOut[] }[] {
  const parts: { name: string; rows: TestRowOut[] }[] = [];
  let previous: number | null = null;

  for (const row of tests) {
    const name = panelOf(row);
    const current = serial(row);
    const last = parts[parts.length - 1];
    const restarted = last !== undefined && previous !== null && current !== null && current <= previous;

    if (last === undefined || last.name !== name || restarted) {
      parts.push({ name, rows: [row] });
    } else {
      last.rows.push(row);
    }
    previous = current;
  }

  // Once a heading covers more than one table, the heading alone no longer
  // identifies which one you are looking at.
  const counts = new Map<string, number>();
  for (const part of parts) counts.set(part.name, (counts.get(part.name) ?? 0) + 1);

  const seen = new Map<string, number>();
  return parts.map((part) => {
    const total = counts.get(part.name) ?? 1;
    if (total < 2) return part;
    const n = (seen.get(part.name) ?? 0) + 1;
    seen.set(part.name, n);
    return { ...part, name: part.name ? `${part.name} · Table ${n} of ${total}` : `Table ${n} of ${total}` };
  });
}

export default function TestResultsTable({ tests }: Props) {
  if (tests.length === 0) return null;

  if (used(tests).length === 0) return null;

  const panels = segment(tests);

  return (
    <div>
      {panels.map((panel, panelIndex) => {
        // Columns are chosen per table, not once for the document: the two
        // tables on one page rarely carry the same ones, and a column that
        // only the other table filled would show as a row of dashes here.
        const columns = used(panel.rows);
        if (columns.length === 0) return null;
        const numeric = numericColumns(panel.rows, columns);

        return (
        <div className="tests-panel" key={`${panel.name}-${panelIndex}`}>
          {panel.name && <h4 className="rg-subheading">{panel.name}</h4>}
          {/* Wide tables scroll inside their own box rather than pushing the
              review column sideways. */}
          <div className="tests-scroll">
            <table className="tests-table">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c} className={numeric.has(c) ? "is-num" : undefined} scope="col">
                      {label(c)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {panel.rows.map((row, i) => {
                  const flag = row.flag ? readFlag(String(row.flag)) : null;
                  const outOfRange = flag?.tone === "high" || flag?.tone === "low";
                  return (
                    <tr key={i} className={outOfRange ? "is-out" : undefined}>
                      {columns.map((c) => {
                        const v = row[c];
                        const text = v === null || v === undefined ? "" : String(v);
                        const classes = [
                          numeric.has(c) ? "is-num" : "",
                          c === "result" ? "tests-result" : "",
                        ]
                          .filter(Boolean)
                          .join(" ");
                        return (
                          <td key={c} className={classes || undefined}>
                            {c === "flag" && flag ? (
                              <span className={`tests-flag tone-${flag.tone}`}>{flag.text}</span>
                            ) : (
                              text || <span className="tests-blank">—</span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
        );
      })}
    </div>
  );
}
