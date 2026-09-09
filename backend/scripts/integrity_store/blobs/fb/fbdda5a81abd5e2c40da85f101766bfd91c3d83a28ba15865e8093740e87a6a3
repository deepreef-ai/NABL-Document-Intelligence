/**
 * Give every extracted field a meaningful section name.
 *
 * "Page 1" tells a reviewer nothing they cannot already see. What they need is
 * the kind of thing a field is — laboratory details, sample details, results —
 * so they can find the value they are looking for and spot the section that is
 * suspiciously empty.
 *
 * This is deterministic keyword classification over the field NAME, not the
 * value, and not a model. That matters: the section label sits next to a value
 * someone is about to confirm, so it must be reproducible and never
 * confidently wrong in a way that changes between runs.
 *
 * Two things make it work on real documents rather than just tidy ones:
 *
 * - **Order is the rule.** `date_of_sample_receipt` matches both "date" and
 *   "sample"; `test_report_status` matches both "test" and "report". The first
 *   category that matches wins, so the categories are ordered by which reading
 *   is right when they overlap.
 * - **A numeric fallback for analytes.** A blood panel extracts as
 *   `bilirubin_total`, `albumin`, `alkaline_phosphatase` — no keyword will ever
 *   enumerate those. A field with no admin keyword and a numeric value is a
 *   measurement, which is a far better guess than "Other".
 */

export const SECTION_ORDER = [
  "Laboratory Details",
  "Patient Details",
  "Customer Details",
  "Report Details",
  "Sample Details",
  "Test Details",
  "Personnel",
  "Equipment",
  "Test Results",
  "Other Details",
] as const;

export type SectionName = (typeof SECTION_ORDER)[number];

/** Checked in order; first match wins. */
const RULES: { section: SectionName; patterns: RegExp }[] = [
  {
    section: "Laboratory Details",
    patterns:
      /(organisation|organization|laboratory_name|lab_name|^lab$|^address|_address|premises|^phone|telephone|^fax|^email|website|^gst|gstin|^pan|^tan|^cin|registration_no|registering|accreditation|nabl_)/,
  },
  {
    section: "Patient Details",
    patterns:
      /(patient|^sex$|gender|^dob$|date_of_birth|^age$|^age_|_age$|mrn|medical_record|lifetime_id|chart_no|^uhid|hospital_no|ward|^bed|clinical)/,
  },
  {
    // "patient" is deliberately NOT here — it used to be, which put every
    // patient field under Customer Details. Patient rules are checked first.
    section: "Customer Details",
    patterns: /(customer|client|recipient|consignee|^to_|billed)/,
  },
  {
    section: "Personnel",
    patterns:
      /(analyst|signator|authorized|authorised|approved_by|reviewed_by|verified_by|checked_by|prepared_by|director|manager|pathologist|physician|clinician|consultant|referring|referred_by|technician|designation|qualification|experience|staff)/,
  },
  {
    section: "Equipment",
    patterns: /(equipment|instrument|calibration|make_model|serial_no|serial_number|^make$|^model$)/,
  },
  {
    section: "Report Details",
    patterns:
      /(report|certificate|^lr_no|^ulr|accession|chart_no|^page|issued|^dated|revision|^version|^status$)/,
  },
  {
    section: "Sample Details",
    patterns:
      /(sample|specimen|animal|breed|species|^drawn|collect|receipt|received|mode_of|quantity|container|condition)/,
  },
  {
    section: "Test Details",
    patterns: /(test_method|^method|technique|protocol|testing|procedure|discipline|parameter)/,
  },
  {
    section: "Test Results",
    patterns:
      /(result|observation|finding|typing|interpretation|conclusion|remark|impression|reference_range|normal_range|^unit|_unit$|value)/,
  },
  // Generic dates LAST, on purpose. "date_of_sample_receipt" and
  // "date_of_testing" belong to their own sections, and a date rule placed
  // any earlier would steal both. What reaches here is an unattached date —
  // the report's own date — which is Report Details.
  {
    section: "Report Details",
    patterns: /(^date$|^date_|_date$|_date_|^day$|^month$|^year$|time$)/,
  },
];

const NUMERIC = /^[<>]?=?\s*-?\d+(?:[.,]\d+)?\s*[%a-zA-Z/µ°]*$/;

// Half a urinalysis panel reports qualitatively, not numerically: glucose NEG,
// ketone NEG, nitrite NEG. Those are results, and a numbers-only fallback
// dumped six analytes into "Other Details" on a real report.
const QUALITATIVE =
  /^(neg|pos|negative|positive|nil|absent|present|trace|normal|abnormal|reactive|non[- ]?reactive|detected|not[- ]detected|clear|turbid|nad)\b/i;

/**
 * Classify one field.
 *
 * `value` is used only for the numeric-analyte fallback, never to override a
 * keyword match — a field called `phone` belongs in Laboratory Details whether
 * or not its value happens to look like a number.
 */
export function sectionFor(fieldPath: string, value: string | null): SectionName {
  const name = fieldPath.toLowerCase();

  // A dotted schema path already states its section: organisation.gst_number.
  const prefix = name.includes(".") ? name.split(".")[0].replace(/\[\d*\]$/, "") : "";
  if (prefix) {
    const mapped = RULES.find((r) => r.patterns.test(prefix));
    if (mapped) return mapped.section;
  }

  for (const rule of RULES) {
    if (rule.patterns.test(name)) return rule.section;
  }

  // An unrecognised name holding a measurement or a qualitative verdict is an
  // analyte, not an unknown.
  if (value != null) {
    const v = value.trim();
    if (NUMERIC.test(v) || QUALITATIVE.test(v)) return "Test Results";
  }

  return "Other Details";
}

/** Sort key so sections read in a sensible document order, not alphabetically. */
export function sectionRank(section: string): number {
  const i = (SECTION_ORDER as readonly string[]).indexOf(section);
  if (i !== -1) return i;
  // A derived record group ("Sample Records") is real content and belongs with
  // the data, not dumped after the catch-all.
  return (SECTION_ORDER as readonly string[]).indexOf("Other Details") - 0.5;
}
