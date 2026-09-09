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

/** Human labels for the gold keys. Keep in step with SECTION_LABELS in structured.py. */
const LABELS: Record<string, string> = {
  document_info: "Document",
  lab_info: "Laboratory",
  client_info: "Client",
  patient_info: "Patient",
  sample_info: "Sample",
  report_info: "Report",
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

/** "lab_panel_results" -> "Lab Panel Results", for gold keys not in LABELS. */
function titleCaseKey(key: string): string {
  return key
    .split("_")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
