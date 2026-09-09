export const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

export type NablFormType =
  | "NABL_151"
  | "NABL_152"
  | "NABL_153"
  | "NABL_153A"
  | "NABL_154"
  | "NABL_155"
  | "NABL_157"
  | "NABL_158"
  | "NABL_159";

export interface Application {
  id: string;
  form_type: NablFormType;
  status: "eligibility" | "unlocked" | "reviewing" | "complete";
  created_at: string;
}

export interface PrerequisiteQuestion {
  id: string;
  text: string;
  help_text: string | null;
}

export interface PrerequisiteAnswer {
  question_id: string;
  satisfied: boolean | null;
  detail: string | null;
}

export interface EligibilityState {
  form_type: string;
  answers: Record<string, PrerequisiteAnswer>;
  all_satisfied: boolean;
  next_question: PrerequisiteQuestion | null;
}

export interface ChatEntry {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ExtractedFieldOut {
  id: string;
  field_path: string;
  value: string | null;
  confidence: number;
  needs_review?: boolean;
  source_page: number | null;
  source_bbox: { x: number; y: number; w: number; h: number } | null;
  /** Sub-heading this field sits under in the source document. */
  section?: string | null;
  /**
   * Gold-dataset section key — "lab_info", "patient_info", "signatories".
   * Chosen by the backend (structured.py's route_field), never here.
   */
  group?: string | null;
  accepted: boolean;
}

/**
 * One row of a results table, in the labelled dataset's shape.
 *
 * Every column is optional because the 53 gold records do not agree on which
 * exist — `reference_range` in one lab's format, `bio_ref_interval` in
 * another's, `test_protocol` in a third's. The table renders whichever columns
 * the document actually carried rather than a fixed set with blanks.
 */
export interface TestRowOut {
  panel_name?: string | null;
  section?: string | null;
  test_name?: string | null;
  result?: string | null;
  unit?: string | null;
  reference_range?: string | null;
  flag?: string | null;
  method?: string | null;
  sample_date?: string | null;
  specimen?: string | null;
  s_no?: string | null;
  page_number?: number | null;
  [column: string]: string | number | null | undefined;
}

export interface DocumentOut {
  id: string;
  application_id: string;
  filename: string;
  content_type: string;
  doc_type: string | null;
  extraction_source: string | null;
  // How many pages the backend actually read and extracted from. There is no
  // page cap any more, so this is what confirms a long scan was read end to
  // end rather than truncated.
  page_count: number | null;
  status: "uploaded" | "processing" | "extracted" | "failed";
  error: string | null;
  /**
   * Result-table rows with their columns intact. A results table is not a
   * list of fields: flattened into scalars, "pH 6.5 5-9" becomes one string
   * with the value and the reference range fused and nothing left to compare.
   */
  tests?: TestRowOut[];
  fields: ExtractedFieldOut[];
}

export interface CompiledFormResponse {
  form_type: string;
  form: Record<string, unknown>;
  confidence_threshold: number;
  documents: (Omit<DocumentOut, "application_id" | "content_type" | "error"> & {
    fields: ExtractedFieldOut[];
  })[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, init);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json() as Promise<T>;
}

/**
 * `skipEligibility` creates the application already unlocked. The upload
 * endpoint rejects anything still in "eligibility" status, and the product no
 * longer shows the wizard that used to clear it, so the one screen the user
 * has could not otherwise accept a file.
 */
export function createApplication(form_type: NablFormType, skipEligibility = false) {
  return request<{ application: Application; state: EligibilityState | null; message: string }>("/applications", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ form_type, skip_eligibility: skipEligibility }),
  });
}

export function getWizardState(applicationId: string) {
  return request<{ application: Application; state: EligibilityState; history: ChatEntry[] }>(
    `/applications/${applicationId}/wizard`
  );
}

export function answerWizardQuestion(applicationId: string, message: string) {
  return request<{ application: Application; state: EligibilityState; reply: string }>(
    `/applications/${applicationId}/wizard/answer`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    }
  );
}

export function uploadDocument(applicationId: string, file: File, script = "english") {
  const form = new FormData();
  form.append("file", file);
  form.append("script", script);
  return request<DocumentOut>(`/applications/${applicationId}/documents`, {
    method: "POST",
    body: form,
  });
}

export function listDocuments(applicationId: string) {
  return request<DocumentOut[]>(`/applications/${applicationId}/documents`);
}

export function reextractDocument(documentId: string, script = "english") {
  const form = new FormData();
  form.append("script", script);
  return request<DocumentOut>(`/documents/${documentId}/reextract`, { method: "POST", body: form });
}

export function renderDocumentUrl(documentId: string, page = 0) {
  return `${API_BASE}/documents/${documentId}/render?page=${page}`;
}

/** The extraction in the labelled dataset's shape, for download or diffing. */
export function structuredJsonUrl(documentId: string) {
  return `${API_BASE}/documents/${documentId}/structured`;
}

export function getCompiledForm(applicationId: string, acceptedOnly = false) {
  return request<CompiledFormResponse>(
    `/applications/${applicationId}/form?accepted_only=${acceptedOnly}`
  );
}

export function patchField(documentId: string, fieldId: string, body: { value?: string; accepted?: boolean }) {
  return request<ExtractedFieldOut>(`/documents/${documentId}/fields/${fieldId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getChatHistory(applicationId: string) {
  return request<ChatEntry[]>(`/applications/${applicationId}/chat`);
}

export function sendChatMessage(applicationId: string, message: string) {
  return request<{ reply: string }>(`/applications/${applicationId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
}
