import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ChatPanel from "../components/ChatPanel";
import DocumentViewer from "../components/DocumentViewer";
import ReviewFieldGroups from "../components/ReviewFieldGroups";
import TestResultsTable from "../components/TestResultsTable";
import {
  getChatHistory,
  getCompiledForm,
  listDocuments,
  patchField,
  reextractDocument,
  sendChatMessage,
  structuredJsonUrl,
  type ChatEntry,
  type CompiledFormResponse,
  type DocumentOut,
} from "../api/client";

function acceptedCount(document: DocumentOut): number {
  return document.fields.filter((f) => f.accepted).length;
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
              {documents.map((d) => (
                <button
                  key={d.id}
                  className={d.id === selectedId ? "active" : ""}
                  onClick={() => {
                    setSelectedId(d.id);
                    setPage(0);
                  }}
                >
                  {d.filename}
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
                    <h2>{selectedDocument.filename}</h2>
                  </div>
                  {selectedDocument.fields.length > 0 && (
                    <span className={`form-completion-badge ${acceptedCount(selectedDocument) === selectedDocument.fields.length ? "complete" : ""}`}>
                      {acceptedCount(selectedDocument)}/{selectedDocument.fields.length} confirmed
                    </span>
                  )}
                  {/* Opens in a tab rather than downloading: the sandbox the
                      review screen runs in blocks script-driven saves, and a
                      reviewer usually wants to look at the JSON before keeping
                      it anyway. */}
                  <a
                    className="form-json-link"
                    href={structuredJsonUrl(selectedDocument.id)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    View JSON
                  </a>
                  <button onClick={() => handleReextract(selectedDocument.id)}>Re-extract</button>
                </div>

                {/* Extraction notes are intentionally not rendered. They are
                    still returned by the API and visible via "View JSON"
                    above; per-field notes stay on the field they describe
                    (see ExtractionGroups). */}

                {/* Above the fields: on a lab report the results ARE the
                    document, and burying them under twenty header fields makes
                    a reviewer scroll past the only part they came to check. */}
                {(selectedDocument.tests?.length ?? 0) > 0 && (
                  <section className="rg-group">
                    <h3 className="rg-heading">Test Results</h3>
                    <div className="rg-body">
                      <TestResultsTable tests={selectedDocument.tests ?? []} />
                    </div>
                  </section>
                )}

                <ReviewFieldGroups
                  fields={selectedDocument.fields}
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

                <p className="form-letterhead-footer">Auto-extracted draft — confirm each value before relying on it.</p>
              </div>
            )}

            {compiled && (
              <details className="compiled-form-preview">
                <summary>Compiled application form (preview)</summary>
                <pre>{JSON.stringify(compiled.form, null, 2)}</pre>
              </details>
            )}
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
