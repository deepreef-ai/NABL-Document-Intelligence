import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import AssistantPanel from "../components/AssistantPanel";
import DocumentViewer from "../components/DocumentViewer";
import { DownloadIcon, FileIcon, PlusIcon, RefreshIcon } from "../components/Icon";
import JsonPanel from "../components/JsonPanel";
import ReviewFieldGroups from "../components/ReviewFieldGroups";
import { splitRedundantFields } from "../api/redundantFields";
import {
  formPdfUrl,
  getChatHistory,
  getCompiledForm,
  getStructured,
  listDocuments,
  patchField,
  reextractDocument,
  sendChatMessage,
  type ChatEntry,
  type CompiledFormResponse,
  type DocumentOut,
} from "../api/client";

/**
 * Steps 2 and 3 — review one document at a time, then approve the lot.
 *
 * There is no header and no sidebar above this, only a tab strip. The screen
 * is a scan on the left and the values read off it on the right, and both
 * want every pixel of width they can get; a persistent frame would buy
 * navigation this flow does not have, at the cost of the thing the flow is
 * for.
 *
 * Which document is on screen lives in the URL (`?doc=`), so the upload
 * page's "Review" links can open a specific one and a reload keeps it.
 */

/**
 * The fields a document actually asks a person to confirm.
 *
 * Not `document.fields`: some of those only repeat a cell of the results
 * table (see splitRedundantFields). They are dropped here rather than inside
 * the form, because everything downstream counts them — the tab pill, the
 * progress bar, the per-section totals and, critically, whether Approve &
 * submit ever enables. A field hidden from the form but still counted is a
 * field nobody can confirm and a button that stays grey forever.
 */
function reviewableFields(document: DocumentOut) {
  return splitRedundantFields(document.fields, document.tests ?? []);
}

/**
 * What to call each document on screen, without using its filename.
 *
 * Uploaded filenames are not neutral: a real one in this project reads
 * "MrYASH_22Y_Male_1026987_20260630164920.PDF", which puts a patient's name,
 * age and sex into the tab, every screenshot and any shared session. The
 * document's classified type says what a reviewer actually needs to know.
 *
 * Numbering runs per TYPE, not per upload: two diagnostic reports beside a
 * calibration certificate are "Diagnostic Report 1" and "Diagnostic Report 2",
 * not 2 and 4. Numbered by position in the upload order, a tab's number moved
 * whenever an unrelated document was removed from the list.
 *
 * A type with only one document is not numbered at all — "Diagnostic Report 1"
 * beside nothing else invites a hunt for report 2.
 */
function documentLabels(documents: DocumentOut[]): Map<string, string> {
  const name = (d: DocumentOut) => {
    const type = (d.doc_type ?? "").trim();
    return type ? titleCaseType(type) : "Document";
  };

  const totals = new Map<string, number>();
  for (const d of documents) totals.set(name(d), (totals.get(name(d)) ?? 0) + 1);

  const seen = new Map<string, number>();
  const labels = new Map<string, string>();
  for (const d of documents) {
    const base = name(d);
    const n = (seen.get(base) ?? 0) + 1;
    seen.set(base, n);
    labels.set(d.id, (totals.get(base) ?? 0) > 1 ? `${base} ${n}` : base);
  }
  return labels;
}

/** "lab_report" / "lab-report" -> "Lab Report". */
function titleCaseType(value: string): string {
  return value
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

/** "NABL_151" -> "NABL 151". */
function formLabel(formType: string | undefined): string {
  return (formType ?? "NABL 151").replace(/_/g, " ");
}

export default function ReviewPage() {
  const { applicationId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [compiled, setCompiled] = useState<CompiledFormResponse | null>(null);
  const [page, setPage] = useState(0);
  const [highlightedFieldId, setHighlightedFieldId] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatEntry[]>([]);
  const [structured, setStructured] = useState<unknown>(null);
  const [structuredError, setStructuredError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedId = searchParams.get("doc");

  const refresh = useCallback(async () => {
    if (!applicationId) return;
    const [docs, form] = await Promise.all([
      listDocuments(applicationId),
      getCompiledForm(applicationId),
    ]);
    setDocuments(docs);
    setCompiled(form);
  }, [applicationId]);

  useEffect(() => {
    refresh().catch((e) => setError(String(e instanceof Error ? e.message : e)));
  }, [refresh]);

  useEffect(() => {
    if (applicationId) {
      getChatHistory(applicationId)
        .then(setChatHistory)
        .catch((e) => setError(String(e instanceof Error ? e.message : e)));
    }
  }, [applicationId]);

  // Nothing selected yet — land on the first document that actually
  // extracted, since an "uploaded" or "failed" one has nothing to review.
  // `replace`, so Back does not walk through every tab that was auto-picked.
  useEffect(() => {
    if (documents.length === 0) return;
    if (selectedId && documents.some((d) => d.id === selectedId)) return;
    const first = documents.find((d) => d.status === "extracted") ?? documents[0];
    setSearchParams({ doc: first.id }, { replace: true });
  }, [documents, selectedId, setSearchParams]);

  const selectedIndex = documents.findIndex((d) => d.id === selectedId);
  const selectedDocument = selectedIndex >= 0 ? documents[selectedIndex] : null;
  const labels = useMemo(() => documentLabels(documents), [documents]);
  const reviewable = useMemo(
    () => new Map(documents.map((d) => [d.id, reviewableFields(d)])),
    [documents],
  );
  const selectedLabel = selectedDocument ? labels.get(selectedDocument.id) ?? "Document" : null;
  const threshold = compiled?.confidence_threshold ?? 0.85;

  // The structured record is per document and unrelated to the field rows, so
  // it is fetched on its own rather than folded into refresh() — a re-extract
  // changes both, a confirm changes neither.
  const selectedDocumentId = selectedDocument?.id;
  useEffect(() => {
    if (!selectedDocumentId) return;
    let stale = false;
    setStructured(null);
    setStructuredError(null);
    getStructured(selectedDocumentId)
      .then((json) => { if (!stale) setStructured(json); })
      .catch(() => {
        if (!stale) {
          setStructuredError(
            "No structured record for this document yet — it was extracted before this format "
            + "existed, or by the legacy pipeline. Re-extract to build it.",
          );
        }
      });
    return () => { stale = true; };
  }, [selectedDocumentId]);

  function selectDocument(id: string) {
    setSearchParams({ doc: id });
    setPage(0);
    setHighlightedFieldId(null);
    setSubmitted(false);
  }

  async function handleSaveField(documentId: string, fieldId: string, value: string) {
    await patchField(documentId, fieldId, { value });
    await refresh();
  }

  async function handleAcceptField(documentId: string, fieldId: string, accepted: boolean) {
    await patchField(documentId, fieldId, { accepted });
    await refresh();
  }

  /**
   * Confirm-all and reset, for the document on screen.
   *
   * Scoped to this document rather than the application: the tab is the unit
   * of review, and a "Reset review" that silently cleared the other
   * document's confirmations would undo work that is not on screen.
   *
   * Only fields that need changing are patched — sending 200 no-op writes to
   * confirm the twelve that are not yet confirmed is a lot of round trips for
   * nothing. One refresh at the end, not one per field.
   */
  async function setAllAccepted(document: DocumentOut, accepted: boolean) {
    // Only the fields the form actually shows. Confirming a folded duplicate
    // would mark something the reviewer was never given a chance to read.
    const shown = reviewable.get(document.id)?.fields ?? document.fields;
    const changing = shown.filter((f) => f.accepted !== accepted);
    if (changing.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      await Promise.all(changing.map((f) => patchField(document.id, f.id, { accepted })));
      await refresh();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  }

  async function handleReextract(documentId: string) {
    setBusy(true);
    setError(null);
    try {
      await reextractDocument(documentId);
      await refresh();
      const json = await getStructured(documentId).catch(() => null);
      setStructured(json);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  }

  async function handleChatSend(message: string) {
    if (!applicationId) return;
    setChatHistory((h) => [...h, { role: "user", content: message }]);
    const res = await sendChatMessage(applicationId, message);
    setChatHistory((h) => [...h, { role: "assistant", content: res.reply }]);
  }

  function exportJson() {
    if (!compiled) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(compiled.form, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${compiled.form_type}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const totals = useMemo(() => {
    const all = documents.flatMap((d) => reviewable.get(d.id)?.fields ?? []);
    return { total: all.length, confirmed: all.filter((f) => f.accepted).length };
  }, [documents, reviewable]);

  const selectedFields = selectedDocument ? reviewable.get(selectedDocument.id) : undefined;
  const docTotal = selectedFields?.fields.length ?? 0;
  const docConfirmed = selectedFields?.fields.filter((f) => f.accepted).length ?? 0;
  const percent = docTotal === 0 ? 0 : Math.round((docConfirmed / docTotal) * 100);

  // Approve & submit covers the whole application, so it waits for every
  // document. Confirming one tab and finding the button still disabled would
  // be baffling on its own, which is what the note beside it is for.
  const outstanding = totals.total - totals.confirmed;
  const canSubmit = totals.total > 0 && outstanding === 0;

  return (
    <div className="review-page">
      <nav className="tabstrip" aria-label="Uploaded documents">
        {documents.map((d) => (
          <button
            key={d.id}
            className={`tab${d.id === selectedId ? " is-active" : ""}`}
            onClick={() => selectDocument(d.id)}
            aria-current={d.id === selectedId ? "page" : undefined}
          >
            {/* Labelled by what the document IS, not what the file was called
                — see documentLabels(). */}
            {labels.get(d.id)}
          </button>
        ))}
        <Link className="tab tab-add" to={`/applications/${applicationId}/upload`}>
          <PlusIcon size={15} />
          Add document
        </Link>

        <div className="tabstrip-meta">
          {selectedDocument && (
            <span className="tabstrip-file" title={selectedDocument.filename}>
              <FileIcon size={14} />
              {selectedDocument.filename}
            </span>
          )}
          <span className={`chip${docTotal > 0 && docConfirmed === docTotal ? " chip-accent" : ""}`}>
            {docConfirmed} / {docTotal} confirmed
          </span>
        </div>
      </nav>

      {error && <p className="error-note" role="alert" style={{ marginTop: 16 }}>{error}</p>}

      <div className="review-grid">
        <div className="viewer-col">
          {selectedDocument ? (
            <DocumentViewer
              document={selectedDocument}
              page={page}
              onPageChange={setPage}
              highlightedFieldId={highlightedFieldId}
            />
          ) : (
            <div className="viewer">
              <p className="viewer-sheet-empty">No document selected.</p>
            </div>
          )}
        </div>

        <div className="form-col">
          <section className="card">
            <div className="autofill-head">
              <h2>{formLabel(compiled?.form_type)} · auto-filled fields</h2>
              <div className="autofill-head-actions">
                <button
                  className="btn"
                  onClick={() => selectedDocument && void handleReextract(selectedDocument.id)}
                  disabled={!selectedDocument || busy}
                >
                  <RefreshIcon size={15} />
                  Re-extract
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => selectedDocument && void setAllAccepted(selectedDocument, true)}
                  disabled={!selectedDocument || busy || docTotal === 0 || docConfirmed === docTotal}
                >
                  Confirm all
                </button>
              </div>
              <div className="progress">
                <div
                  className="progress-track"
                  role="progressbar"
                  aria-valuenow={percent}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label="Fields confirmed on this document"
                >
                  <span style={{ width: `${percent}%` }} />
                </div>
                <span className="progress-label">{percent}%</span>
              </div>
            </div>

            {selectedDocument ? (
              <ReviewFieldGroups
                fields={selectedFields?.fields ?? selectedDocument.fields}
                tests={selectedDocument.tests ?? []}
                foldedCount={selectedFields?.folded.length ?? 0}
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
            ) : (
              <p className="rg-empty" style={{ padding: 16 }}>
                Upload a document to begin auto-filling the {formLabel(compiled?.form_type)} form.
              </p>
            )}
          </section>
        </div>
      </div>

      <JsonPanel
        data={structured}
        schema={`${formLabel(compiled?.form_type)} schema`}
        filename={selectedLabel ? selectedLabel.replace(/\s+/g, "-").toLowerCase() : "extraction"}
        fieldCount={selectedDocument?.fields.length ?? 0}
        testCount={selectedDocument?.tests?.length ?? 0}
        error={structuredError}
      />

      <div className="submit-row">
        <p className="submit-note">
          {submitted
            ? "Approved. The compiled form opened in a new tab — every value in it carries the page it was read from."
            : canSubmit
              ? `All ${totals.total} extracted values are confirmed across ${documents.length} document${documents.length === 1 ? "" : "s"}.`
              : `${outstanding} value${outstanding === 1 ? "" : "s"} still to confirm across ${documents.length} document${documents.length === 1 ? "" : "s"}. Auto-extracted values are a draft until a person confirms them.`}
        </p>
        <div className="submit-actions">
          <button
            className="btn"
            onClick={() => selectedDocument && void setAllAccepted(selectedDocument, false)}
            disabled={!selectedDocument || busy || docConfirmed === 0}
          >
            Reset review
          </button>
          <button className="btn" onClick={exportJson} disabled={!compiled}>
            <DownloadIcon size={15} />
            Export JSON
          </button>
          <button
            className="btn btn-primary"
            disabled={!canSubmit || busy}
            onClick={() => {
              window.open(formPdfUrl(applicationId!), "_blank", "noopener");
              setSubmitted(true);
            }}
          >
            Approve &amp; submit
          </button>
        </div>
      </div>

      <AssistantPanel
        history={chatHistory}
        onSend={handleChatSend}
        documentLabel={selectedLabel ? selectedLabel.toLowerCase() : "this document"}
      />
    </div>
  );
}
