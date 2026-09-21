/**
 * The gold dataset's section taxonomy, mirrored for display.
 *
 * The backend decides which section a field belongs to — one place,
 * app/graph/structured.py's `route_field` — and sends the answer as
 * `field.group`. This module only names and orders those sections for the
 * screen; it never classifies.
 *
 * That split is the point. The frontend used to run its own keyword classifier
 * over field names, so the backend could file a value under `lab_info` in the
 * exported JSON while the form showed it under "Other Details". Same document,
 * two answers, and nothing to say which was right. There is one taxonomy now,
 * it is the labelled dataset's, and this file is downstream of it.
 */

/**
 * Human labels for the gold keys. Keep in step with SECTION_LABELS in
 * structured.py.
 *
 * The `*_info` sections read as "<thing> Details": a heading of just
 * "Client" names the party, where what follows it is the set of details
 * ABOUT that party, and on a page of stacked headings the bare nouns read as
 * labels for a single value rather than as section titles.
 *
 * The rest keep their own names. "Findings", "Signatories", "Notes" and
 * "Test Results" already describe their contents, so "Notes Details" would
 * be padding.
 */
const LABELS: Record<string, string> = {
  document_info: "Document Details",
  lab_info: "Laboratory Details",
  client_info: "Client Details",
  patient_info: "Patient Details",
  sample_info: "Lab Sample Details",
  report_info: "Report Details",
  findings: "Findings",
  signatories: "Signatories",
  notes: "Notes",
  tests: "Test Results",
};

/**
 * Reading order, as the gold records are written: who ran the test, for whom,
 * on what, and what came back — then the findings, the signatures and the
 * small print.
 */
const ORDER = [
  "document_info",
  "lab_info",
  "client_info",
  "patient_info",
  "sample_info",
  "report_info",
  "findings",
  "tests",
  "signatories",
  "notes",
];

const UNFILED = "Other";

export function goldLabel(group: string | null | undefined): string {
  if (!group) return UNFILED;
  return LABELS[group] ?? titleCaseKey(group);
}

export function goldRank(label: string): number {
  const index = ORDER.findIndex((key) => goldLabel(key) === label);
  // Anything the backend routed somewhere this file has no name for sorts
  // last rather than first — an unrecognised section is the least likely to
  // be what a reviewer opened the page to check.
  return index === -1 ? ORDER.length : index;
}

/**
 * "lab_panel_results" -> "Lab Panel Results", for gold keys not in LABELS.
 *
 * A key the backend adds later that follows the `*_info` convention gets the
 * same "<thing> Details" treatment as the ones named above, so a new section
 * does not arrive on screen reading "Equipment Info" beside "Client Details".
 */
function titleCaseKey(key: string): string {
  const words = key.split("_").filter(Boolean);
  if (words.length > 1 && words[words.length - 1] === "info") {
    words[words.length - 1] = "details";
  }
  return words.map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}
