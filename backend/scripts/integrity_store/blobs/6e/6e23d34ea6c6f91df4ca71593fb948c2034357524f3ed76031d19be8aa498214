/**
 * Client for the agentic extraction workflow (/graph/*).
 *
 * Kept separate from client.ts on purpose: the legacy endpoints speak the old
 * flattened document/field shape, and this one speaks the graph's richer
 * contract — evidence status, preserved conflict candidates, per-page
 * coverage, resumable runs. Merging the two type sets would mean one union
 * type where half the fields are always undefined.
 */

export const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

// ---------------------------------------------------------------- types

export type ExtractionStatus = "VERIFIED" | "UNCERTAIN" | "CONFLICTED" | "NOT_FOUND";

export type EvidenceStatus =
  | "SUPPORTED"
  | "PARTIALLY_SUPPORTED"
  | "UNSUPPORTED"
  | "WRONG_PAGE"
  | "NO_EVIDENCE";

export type MappingStatus =
  | "MAPPED"
  | "PARTIALLY_MAPPED"
  | "NOT_FOUND"
  | "AMBIGUOUS"
  | "CONFLICTED"
  | "MANUAL_REVIEW_REQUIRED";

export type FinalStatus =
  | "VALIDATED"
  | "INCOMPLETE"
  | "LOW_CONFIDENCE"
  | "CONFLICTED"
  | "MANUAL_REVIEW_REQUIRED"
  | "REJECTED";

export interface GraphField {
  field_name: string;
  normalized_field_name: string;
  value: string | null;
  normalized_value: string | null;
  data_type: string;
  page_number: number | null;
  chunk_id: string;
  /** The heading this field sits under — the grouping key in the UI. */
  section_name: string;
  /** Row identity when the field came out of a table. */
  table_context: string;
  exact_source_evidence: string;
  confidence_score: number;
  extraction_status: ExtractionStatus;
  evidence_status: EvidenceStatus | null;
  occurrence_index: number;
  merged_from: string[];
  field_uid: string;
  notes: string;
}

export interface ConflictCandidate {
  field_uid: string;
  value: string | null;
  page_number: number | null;
  chunk_id: string;
  section_name: string;
  confidence_score: number;
  evidence: string;
}

export interface GraphConflict {
  normalized_field_name: string;
  candidates: ConflictCandidate[];
  resolution: "RESOLVED" | "UNRESOLVED" | "MANUAL_REVIEW_REQUIRED";
  chosen_field_uid: string | null;
  reason: string;
  criteria_used: string[];
}

export interface GraphMapping {
  source_field: string;
  target_field: string;
  mapped_value: string | null;
  mapping_confidence: number;
  source_page: number | null;
  source_evidence: string;
  mapping_reason: string;
  mapping_status: MappingStatus;
  source_field_uid: string;
}

export interface ManualReviewItem {
  kind: string;
  reference: string;
  summary: string;
  page_number: number | null;
  candidates: unknown[];
  suggested_action: string;
}

export interface QualityControl {
  all_pages_processed: boolean;
  all_chunks_processed: boolean;
  all_values_have_evidence: boolean;
  conflicts_resolved: boolean;
  form_mapping_verified: boolean;
  information_loss_detected: boolean;
}

export interface DocumentSummary {
  document_type: string | null;
  page_count: number;
  processed_pages: number;
  processed_chunks: number;
  failed_chunks: number;
  ocr_pages: number;
}

export interface AuditEntry {
  timestamp: string;
  node: string;
  event: string;
  detail: string;
  page_number: number | null;
  chunk_id: string | null;
  level: "info" | "warning" | "error";
}

export interface GraphResult {
  overall_status: FinalStatus;
  document_summary: DocumentSummary;
  extracted_fields: GraphField[];
  form_mappings: GraphMapping[];
  filled_form: Record<string, unknown>;
  missing_fields: string[];
  conflicts: GraphConflict[];
  failed_chunks: string[];
  manual_review_items: ManualReviewItem[];
  quality_control: QualityControl;
  final_reasoning: string;
  audit_log: AuditEntry[];
  metrics: {
    llm_call_count: number;
    retry_count: number;
    processing_time_seconds: number;
    token_usage: Record<string, number>;
  };
}

export interface GraphStatus {
  source: "checkpoint" | "database";
  document_id: string;
  current_node?: string;
  current_status?: string;
  final_status?: FinalStatus | null;
  awaiting_human?: boolean;
  next_nodes?: string[];
  total_pages?: number;
  chunks_total?: number;
  chunks_processed?: number;
  chunks_failed?: number;
  fields_extracted?: number;
  llm_calls?: number;
  manual_review_items?: number;
}

export interface RunSummary {
  document_id: string;
  file_name: string;
  final_status: string;
  current_status: string;
  awaiting_human: boolean;
  total_pages: number;
  failed_chunks: number;
  llm_call_count: number;
  processing_time_seconds: number;
  created_at: string | null;
}

// ---------------------------------------------------------------- transport

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    // FastAPI puts the useful part in `detail`; surfacing the raw status alone
    // gives the user nothing to act on.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

// ---------------------------------------------------------------- endpoints

export async function submitDocument(
  file: File,
  options: { formId?: string; targetFormSchema?: unknown } = {},
): Promise<{ document_id: string; status: string; poll: string }> {
  const body = new FormData();
  body.append("file", file);
  if (options.formId) body.append("form_id", options.formId);
  if (options.targetFormSchema) body.append("target_form_schema", JSON.stringify(options.targetFormSchema));
  return request(`/graph/documents`, { method: "POST", body });
}

export function getStatus(documentId: string): Promise<GraphStatus> {
  return request(`/graph/documents/${encodeURIComponent(documentId)}/status`);
}

export function getResult(documentId: string): Promise<GraphResult> {
  return request(`/graph/documents/${encodeURIComponent(documentId)}/result`);
}

export function listRuns(limit = 25): Promise<{ count: number; runs: RunSummary[] }> {
  return request(`/graph/documents?limit=${limit}`);
}

export function submitReview(
  documentId: string,
  decisions: { field_name: string; chosen_field_uid: string }[],
): Promise<GraphResult> {
  return request(`/graph/documents/${encodeURIComponent(documentId)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decisions }),
  });
}

export function retryDocument(documentId: string): Promise<GraphResult> {
  return request(`/graph/documents/${encodeURIComponent(documentId)}/retry`, { method: "POST" });
}

// ---------------------------------------------------------------- grouping

export interface FieldRow {
  field: GraphField;
  /** True when this row is one of several sharing a table_context. */
  isTableRow: boolean;
}

export interface FieldGroup {
  /** The heading shown in the UI. */
  heading: string;
  /** Rows directly under the heading. */
  rows: FieldRow[];
  /** Sub-tables inside this heading, keyed by table_context. */
  tables: { context: string; rows: FieldRow[] }[];
  pages: number[];
  verified: number;
  total: number;
}

const UNGROUPED = "Other fields";

/**
 * Group extracted fields under their heading, and table rows under their table.
 *
 * This is the shape the review screen is built around: a reader scanning a lab
 * report thinks in headings — "Results", "Sample Details" — and then the tests
 * under each. A flat list of forty fields loses that structure, which is
 * exactly the structure the document had and the reason the values make sense
 * together.
 *
 * `section_name` is the heading. `table_context` distinguishes rows inside a
 * repeated table, so twelve equipment rows read as twelve rows of one table
 * rather than twelve unrelated fields that happen to share a name.
 */
export function groupFields(fields: GraphField[]): FieldGroup[] {
  const bySection = new Map<string, GraphField[]>();

  for (const f of fields) {
    const heading = (f.section_name || "").trim() || UNGROUPED;
    const bucket = bySection.get(heading);
    if (bucket) bucket.push(f);
    else bySection.set(heading, [f]);
  }

  const groups: FieldGroup[] = [];

  for (const [heading, items] of bySection) {
    const tableBuckets = new Map<string, GraphField[]>();
    const loose: GraphField[] = [];

    for (const f of items) {
      const ctx = (f.table_context || "").trim();
      if (ctx) {
        const bucket = tableBuckets.get(ctx);
        if (bucket) bucket.push(f);
        else tableBuckets.set(ctx, [f]);
      } else {
        loose.push(f);
      }
    }

    const pages = [...new Set(items.map((f) => f.page_number).filter((p): p is number => p != null))].sort(
      (a, b) => a - b,
    );

    groups.push({
      heading,
      rows: loose.map((field) => ({ field, isTableRow: false })),
      tables: [...tableBuckets.entries()].map(([context, rows]) => ({
        context,
        rows: rows.map((field) => ({ field, isTableRow: true })),
      })),
      pages,
      verified: items.filter((f) => f.extraction_status === "VERIFIED").length,
      total: items.length,
    });
  }

  // Document order, by the first page each heading appears on. Alphabetical
  // would scramble a report whose sections have a meaningful sequence.
  groups.sort((a, b) => {
    const pa = a.pages[0] ?? Number.MAX_SAFE_INTEGER;
    const pb = b.pages[0] ?? Number.MAX_SAFE_INTEGER;
    if (pa !== pb) return pa - pb;
    if (a.heading === UNGROUPED) return 1;
    if (b.heading === UNGROUPED) return -1;
    return a.heading.localeCompare(b.heading);
  });

  return groups;
}
