import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ChatPanel from "../components/ChatPanel";
import DocumentViewer from "../components/DocumentViewer";
import ReviewFieldGroups from "../components/ReviewFieldGroups";
import TestResultsTable from "../components/TestResultsTable";
import { dedupeRepeatedReadings, splitRedundantFields } from "../api/redundantFields";
import {
  getChatHistory,
  getCompiledForm,
  listDocuments,
  patchField,
  reextractDocument,
  sendChatMessage,
  type ChatEntry,
  type CompiledFormResponse,
  type DocumentOut,
  type ExtractedFieldOut,
} from "../api/client";

/**
 * What to call a document on screen, without using its filename.
 *
 * Uploaded filenames are not neutral: a real one in this project reads
 * "MrYASH_22Y_Male_1026987_20260630164920.PDF", which puts a patient's name,
 * age and sex into the tab, the heading, every screenshot and any shared
 * session. The document's classified type says what a reviewer needs.
 *
 * The index disambiguates two documents of the same type so tabs stay
 * distinguishable; it is 1-based because it is a label, not an offset.
 */
function documentLabel(document: DocumentOut, index: number): string {
  const type = (document.doc_type ?? "").trim();
  const name = type ? titleCaseType(type) : "Document";
  return index >= 0 ? `${name} ${index + 1}` : name;
}

/** "lab_report" / "lab-report" -> "Lab Report". */
function titleCaseType(value: string): string {
  return value
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

/**
 * The fields a document actually asks a person to confirm.
 *
 * Not `document.fields`: a lab report prints each analyte's method inside
 * the results table, and extraction picks it up twice — once as the table's
 * Method column and once as a loose "METHOD : BIURATE" pair. Showing both
 * gives the reviewer a row of cards repeating the strings already in the
 * column above, each with its own tick. splitRedundantFields folds those
 * away, and only when the value really is in this document's table.
 */
function reviewableFields(document: DocumentOut): ExtractedFieldOut[] {
  // Two folds, in order: drop methods the results table already prints, then
  // collapse a path re-read off page furniture (a letterhead repeated on
  // every page yields the same ISO line once per page).
  const withoutTableMethods = splitRedundantFields(document.fields, document.tests ?? []).fields;
  return dedupeRepeatedReadings(withoutTableMethods).fields;
}

function acceptedCount(document: DocumentOut): number {
  // Counted over the same set the form shows, so "n/n confirmed" can
  // actually reach its total — a hidden field nobody can tick would
  // otherwise hold the count below 100% forever.
  return reviewableFields(document).filter((f) => f.accepted).length;
}

export default function ReviewPage() {
  const { applicationId } = useParams();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [compiled, setCompiled] = useState<CompiledFormResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [highlightedFieldId, setHighlightedFieldId] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatEntry[]>([]);
  const [chatOpen, setChatOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!applicationId) return;
    const [docs, form] = await Promise.all([listDocuments(applicationId), getCompiledForm(applicationId)]);
    setDocuments(docs);
    setCompiled(form);
    setSelectedId((current) => current ?? docs.find((d) => d.status === "extracted")?.id ?? docs[0]?.id ?? null);
  }

  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
  }, [applicationId]);

  useEffect(() => {
    if (applicationId) {
      getChatHistory(applicationId).then(setChatHistory).catch((e) => setError(String(e)));
    }
  }, [applicationId]);

  const selectedDocument = documents.find((d) => d.id === selectedId) ?? null;
  const threshold = compiled?.confidence_threshold ?? 0.85;

  async function handleSaveField(documentId: string, fieldId: string, value: string) {
    await patchField(documentId, fieldId, { value });
    await refresh();
  }

  async function handleAcceptField(documentId: string, fieldId: string, accepted: boolean) {
    await patchField(documentId, fieldId, { accepted });
    await refresh();
  }

  async function handleReextract(documentId: string) {
    await reextractDocument(documentId);
    await refresh();
  }

  async function handleChatSend(message: string) {
    if (!applicationId) return;
    setChatHistory((h) => [...h, { role: "user", content: message }]);
    const res = await sendChatMessage(applicationId, message);
    setChatHistory((h) => [...h, { role: "assistant", content: res.reply }]);
  }

  return (
    <div className="page review-page">
      <div className="review-header">
        <h1>Review & auto-fill — {compiled?.form_type}</h1>
      </div>
      {error && <p className="error">{error}</p>}

      <div className="review-workspace">
        <div className="review-content-column">
          <div className="review-document-column">
            <div className="document-tabs">
              {documents.map((d, i) => (
                <button
                  key={d.id}
                  className={d.id === selectedId ? "active" : ""}
                  onClick={() => {
                    setSelectedId(d.id);
                    setPage(0);
                  }}
                >
                  {/* The type, not the filename — see documentLabel(). */}
                  {documentLabel(d, i)}
                </button>
              ))}
            </div>
            {selectedDocument && (
              <DocumentViewer
                document={selectedDocument}
                page={page}
                onPageChange={setPage}
                highlightedFieldId={highlightedFieldId}
              />
            )}
          </div>

          <div className="review-form-column">
            {!selectedDocument && (
              <div className="form-empty-state">
                <p>Upload a document to begin auto-filling the application form.</p>
              </div>
            )}

            {selectedDocument && (
              <div className="form-letterhead">
                <div className="form-letterhead-header">
                  <div>
                    <p className="form-letterhead-eyebrow">
                      {selectedDocument.doc_type ?? "unclassified"} · via {selectedDocument.extraction_source ?? "n/a"}
                      {selectedDocument.page_count !== null && ` · ${selectedDocument.page_count} page${selectedDocument.page_count === 1 ? "" : "s"} read`}
                    </p>
                    {/* The eyebrow above already carries the type, source
                        and page count, so dropping the filename loses no
                        information a reviewer needs while keeping
                        identifying data off the screen. */}
                    <h2>{documentLabel(selectedDocument, documents.indexOf(selectedDocument))}</h2>
                  </div>
                  {reviewableFields(selectedDocument).length > 0 && (
                    <span className={`form-completion-badge ${acceptedCount(selectedDocument) === reviewableFields(selectedDocument).length ? "complete" : ""}`}>
                      {acceptedCount(selectedDocument)}/{reviewableFields(selectedDocument).length} confirmed
                    </span>
                  )}
                  {/* No "View JSON" link. A reviewer confirms values against
                      the page beside them, in the field rows below; a raw
                      dump is a second, unreviewable copy of the same data.
                      The endpoint still exists for anything that needs the
                      machine-readable shape. */}
                  <button onClick={() => handleReextract(selectedDocument.id)}>Re-extract</button>
                </div>

                {/* Extraction notes are intentionally not rendered. They are
                    still returned by the API (the structured endpoint is
                    unchanged, only the on-screen link to it is gone);
                    per-field notes stay on the field they describe (see
                    ExtractionGroups). */}

                <ReviewFieldGroups
                  fields={reviewableFields(selectedDocument)}
                  threshold={threshold}
                  highlightedFieldId={highlightedFieldId}
                  onFocus={setHighlightedFieldId}
                  onJumpToPage={(f) => {
                    if (f.source_page !== null) setPage(f.source_page);
                    setHighlightedFieldId(f.id);
                  }}
                  onSave={(fieldId, value) => handleSaveField(selectedDocument.id, fieldId, value)}
                  onAccept={(f) => handleAcceptField(selectedDocument.id, f.id, !f.accepted)}
                />

                {/* Under the field sections, not pinned above them. It used
                    to be pinned, on the reasoning that on a lab report the
                    results ARE the document — but the form is read against
                    the scan beside it, and a document that opens with its
                    report number and date should open with "Report" here
                    too. Pinning the table put it above every section no
                    matter where the page printed it. */}
                {(selectedDocument.tests?.length ?? 0) > 0 && (
                  <section className="rg-group">
                    <h3 className="rg-heading">Test Results</h3>
                    <div className="rg-body">
                      <TestResultsTable tests={selectedDocument.tests ?? []} />
                    </div>
                  </section>
                )}

                <p className="form-letterhead-footer">Auto-extracted draft — confirm each value before relying on it.</p>
              </div>
            )}

            {/* The compiled form is deliberately not dumped as JSON here.
                It is mostly nulls until every document is reviewed, so it
                read as a wall of empty keys directly under the values a
                reviewer had just confirmed. */}
          </div>
        </div>
      </div>

      <button
        className={`chat-fab ${chatOpen ? "active" : ""}`}
        onClick={() => setChatOpen((open) => !open)}
        aria-label={chatOpen ? "Close chat" : "Open chat"}
      >
        {chatOpen ? (
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 5h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H9l-5 4v-4H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      {chatOpen && (
        <div className="chat-drawer">
          <ChatPanel
            history={chatHistory}
            onSend={handleChatSend}
            placeholder="Ask about an extracted value…"
            title="NABL Assistant"
            subtitle="Ask questions, or request a re-extraction"
          />
        </div>
      )}
    </div>
  );
}
