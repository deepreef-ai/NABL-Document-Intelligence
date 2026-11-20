import type { TestRowOut } from "../api/client";

/**
 * The results table, as a table.
 *
 * A results table is not a list of fields, and treating it as one is the
 * single biggest way an extraction can be technically complete and practically
 * useless. Flattened into scalars, a urine panel's `pH 6.5 5-9` becomes one
 * string with the reading and the reference range fused together: nothing to
 * sort by, nothing to compare, no way to see at a glance that a result sits
 * outside its range. The labelled dataset keeps the columns separate, and so
 * does this.
 *
 * Columns are chosen from the data rather than fixed, because the 53 gold
 * records do not agree on which exist — `reference_range` in one lab's format,
 * `bio_ref_interval` in another's, `test_protocol` in a third's. A fixed set
 * would show four empty columns on most documents and silently drop whatever
 * the fifth lab happened to print.
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

/** "URINE CHEMISTRY" + "Qualitative" -> one heading, skipping either if absent. */
function panelOf(row: TestRowOut): string {
  return [row.panel_name, row.section].filter((p) => p && String(p).trim()).join(" · ");
}

export default function TestResultsTable({ tests }: Props) {
  if (tests.length === 0) return null;

  const columns = used(tests);
  if (columns.length === 0) return null;

  // Preserve document order within a panel, and panel order by first
  // appearance — the order the page printed, which is the order a reviewer
  // checking against the PDF will read.
  const panels: { name: string; rows: TestRowOut[] }[] = [];
  for (const row of tests) {
    const name = panelOf(row);
    const existing = panels.find((p) => p.name === name);
    if (existing) existing.rows.push(row);
    else panels.push({ name, rows: [row] });
  }

  return (
    <div className="tests">
      {panels.map((panel) => (
        <div className="tests-panel" key={panel.name || "__unpanelled__"}>
          {panel.name && <h4 className="rg-subheading">{panel.name}</h4>}
          {/* Wide tables scroll inside their own box rather than pushing the
              review column sideways. */}
          <div className="tests-scroll">
            <table className="tests-table">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c} className={c === "result" ? "tests-result" : undefined}>
                      {label(c)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {panel.rows.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => {
                      const v = row[c];
                      const text = v === null || v === undefined ? "" : String(v);
                      return (
                        <td key={c} className={c === "result" ? "tests-result" : undefined}>
                          {text || <span className="tests-blank">—</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
