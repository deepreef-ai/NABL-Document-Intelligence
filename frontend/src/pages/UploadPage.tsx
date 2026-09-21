import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  createApplication,
  deleteDocument,
  type DocumentOut,
  type DocumentStatus,
  getDocumentStatus,
  listDocuments,
  type NablFormType,
  renderDocumentUrl,
  uploadDocument,
} from "../api/client";
import { CheckIcon, CloseIcon, FileIcon, GaugeIcon, ShieldIcon, UploadIcon } from "../components/Icon";

/**
 * Step 1 — upload.
 *
 * There is no wizard and no start step. Landing here with no application in
 * the URL silently gets one and shows the dropzone; the application is
 * created ALREADY UNLOCKED, which is load-bearing, because the upload
 * endpoint refuses anything still in "eligibility" status and that is the
 * gate the wizard used to satisfy.
 *
 * The URL is the source of truth for which application this is. "/" always
 * starts a fresh one; "/applications/<id>/upload" carries its own id, so a
 * reload still shows exactly the documents belonging to it. Remembering the
 * last application instead used to leave the previous file sitting on screen
 * when someone came back to add a new one, which read as though the upload
 * had not worked.
 */

const SCRIPTS = [
  { value: "english", label: "English (Latin)" },
  { value: "devanagari", label: "Devanagari" },
  { value: "arabic", label: "Arabic" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "ka", label: "Kannada" },
];

const CHECKLIST = [
  "Legal proofs — GST certificate, CIN or registration document",
  "Equipment calibration certificates, in date",
  "Personnel CVs and competence records for signatories",
  "PT / ILC participation results for the scope claimed",
  "SOP excerpts covering each test method listed",
];

const ACCEPTED = ".pdf,.docx,.jpg,.jpeg,.png";

//: Which NABL form the compiled output targets. Not asked for on screen —
//: there is no "start" step — so it lives here until the product needs a
//: picker again.
const DEFAULT_FORM_TYPE: NablFormType = "NABL_151";

const POLL_INTERVAL_MS = 2000;
//: A 17-page PDF measured over 20 minutes end to end, so the ceiling is well
//: clear of that. Reaching it means "still running", not "failed".
const POLL_TIMEOUT_MS = 30 * 60 * 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** "diagnostic_report" -> "Diagnostic report". */
function prettyType(value: string | null): string {
  if (!value) return "Unclassified";
  const words = value.replace(/[_-]+/g, " ").trim().toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function Stepper({ current }: { current: 1 | 2 | 3 }) {
  const steps = ["Upload", "Review & auto-fill", "Approve & submit"];
  return (
    <ol className="stepper">
      {steps.map((label, i) => (
        <li key={label} className={i + 1 === current ? "is-current" : undefined}>
          <span className="stepper-num">{i + 1}</span>
          {label}
        </li>
      ))}
    </ol>
  );
}

export default function UploadPage() {
  const { applicationId } = useParams();
  const navigate = useNavigate();
  const [starting, setStarting] = useState(false);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [script, setScript] = useState("english");
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, DocumentStatus>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Set when the component unmounts, so a poll loop started by this page does
  // not keep running — and keep calling setState — after the user navigates
  // to the review screen.
  const cancelledRef = useRef(false);

  useEffect(() => () => { cancelledRef.current = true; }, []);

  const refresh = useCallback(() => {
    if (!applicationId) return;
    listDocuments(applicationId)
      .then(setDocuments)
      .catch((e) => setError(String(e instanceof Error ? e.message : e)));
  }, [applicationId]);

  useEffect(refresh, [refresh]);

  const start = useCallback(async () => {
    setStarting(true);
    setError(null);
    try {
      const res = await createApplication(DEFAULT_FORM_TYPE, true);
      // `replace`, so Back from here does not land on "/" and immediately
      // create yet another application.
      navigate(`/applications/${res.application.id}/upload`, { replace: true });
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setStarting(false);
    }
  }, [navigate]);

  useEffect(() => {
    if (!applicationId && !starting && !error) {
      void start();
    }
    // `starting`/`error` deliberately excluded: this must fire once, not
    // retry in a loop when creation fails.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applicationId, start]);

  /**
   * Wait for one document to finish extracting.
   *
   * Upload returns 202 the moment the row exists and the work happens after
   * the response, so this polls the light status endpoint rather than
   * re-listing every document and its fields on each tick.
   *
   * Bounded: a stuck job must not poll forever. On timeout the document is
   * left in whatever state the server reports — it is still processing, and
   * reloading will pick it up — rather than being shown as failed.
   */
  const waitForExtraction = useCallback(async (documentId: string) => {
    const deadline = Date.now() + POLL_TIMEOUT_MS;

    while (Date.now() < deadline) {
      if (cancelledRef.current) return;
      let status: DocumentStatus;
      try {
        status = await getDocumentStatus(documentId);
      } catch {
        // A single failed poll is not a failed job — the server may be busy
        // with the very extraction being polled. Try again on the next tick.
        await sleep(POLL_INTERVAL_MS);
        continue;
      }

      setProgress((current) => ({ ...current, [documentId]: status }));
      if (status.done) return;
      await sleep(POLL_INTERVAL_MS);
    }

    setProgress((current) => ({
      ...current,
      [documentId]: {
        ...(current[documentId] ?? {
          document_id: documentId, status: "processing", doc_type: null,
          page_count: null, field_count: 0, error: null,
        }),
        done: false,
        // Deliberately does NOT claim the document is fine and that reloading
        // will pick it up. Extraction runs in the API process, so a restart
        // or a crash ends it silently and the row sits at "processing"
        // forever; reloading then shows the same thing again. Re-extract is
        // the action that actually recovers it, so that is what this says.
        error: `Still running after ${Math.round(POLL_TIMEOUT_MS / 60000)} minutes. `
             + "It may still be working, or the server may have restarted and ended it. "
             + "Open Review and use Re-extract to run it again.",
      },
    }));
  }, []);

  const handleFiles = useCallback(async (files: FileList | File[] | null) => {
    const list = files ? Array.from(files) : [];
    if (!applicationId || list.length === 0) return;
    // Guards against a second submission while one is in flight. The input is
    // also disabled, but that alone does not survive a drop or a fast
    // double-select, and a duplicate upload costs a full extraction.
    if (uploading) return;

    setUploading(true);
    setError(null);
    cancelledRef.current = false;
    try {
      for (const file of list) {
        const doc = await uploadDocument(applicationId, file, script);
        setDocuments((docs) => [...docs.filter((d) => d.id !== doc.id), doc]);
        // Sequential on purpose: each extraction is CPU- and LLM-bound, so
        // starting several at once makes them all slower and increases the
        // chance of provider throttling.
        await waitForExtraction(doc.id);
        refresh();
      }
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [applicationId, refresh, script, uploading, waitForExtraction]);

  async function handleRemove(document: DocumentOut) {
    setError(null);
    try {
      await deleteDocument(document.id);
      setDocuments((docs) => docs.filter((d) => d.id !== document.id));
      setProgress(({ [document.id]: _removed, ...rest }) => rest);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }

  // ---- no application in the URL: get one and carry on -------------------
  if (!applicationId) {
    return (
      <div className="upload-page">
        <div className="upload-head">
          <div>
            <p className="micro">NABL · step 1 of 3</p>
            <h1>Upload supporting documents</h1>
            <p>{error ?? "Preparing your application…"}</p>
          </div>
        </div>
        {error && (
          <button className="btn btn-primary" onClick={start} disabled={starting}>
            Try again
          </button>
        )}
      </div>
    );
  }

  const extracted = documents.filter((d) => d.status === "extracted");
  const totalFields = extracted.reduce((n, d) => n + d.fields.length, 0);
  const confidences = extracted.flatMap((d) => d.fields.map((f) => f.confidence));
  const avgConfidence = confidences.length
    ? confidences.reduce((a, c) => a + c, 0) / confidences.length
    : null;

  return (
    <div className="upload-page">
      <header className="upload-head">
        <div>
          <p className="micro">NABL 151 · step 1 of 3</p>
          <h1>Upload supporting documents</h1>
          <p>
            Add the legal proofs (GST/CIN), equipment calibration certificates, personnel CVs,
            PT/ILC results and SOP excerpts that support the scope you are applying for — each one
            is read, classified and auto-filled into the form on the next step.
          </p>
        </div>
        <Stepper current={1} />
      </header>

      {error && <p className="error-note" role="alert">{error}</p>}

      <div className="upload-grid">
        <div className="upload-col upload-add">
          <section className="card">
            <div className="card-head">
              <h2>Add files</h2>
              {uploading && (
                <span className="chip" role="status" aria-live="polite">
                  Extracting — this can take several minutes
                </span>
              )}
            </div>
            <div className="card-body">
              <div className="script-field">
                <label className="micro" htmlFor="script-select">
                  Script of scanned or photographed pages
                </label>
                <select
                  id="script-select"
                  className="field-select"
                  value={script}
                  onChange={(e) => setScript(e.target.value)}
                  disabled={uploading}
                >
                  {SCRIPTS.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* A label wrapping a visually hidden file input, not a div with
                  a click handler: the input keeps the keyboard focus, the file
                  dialog and its own accessible name. */}
              <div
                className={`dropzone${dragging ? " is-over" : ""}${uploading ? " is-busy" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!uploading) setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  void handleFiles(e.dataTransfer.files);
                }}
              >
                <span className="dropzone-icon">
                  <UploadIcon size={20} />
                </span>
                <p className="dropzone-title">Drag and drop files here</p>
                <p className="dropzone-hint">PDF, DOCX, JPG or PNG · up to 25 MB per file · multi-page scans welcome</p>
                <label className="dropzone-cta">
                  <input
                    ref={fileInputRef}
                    className="visually-hidden"
                    type="file"
                    multiple
                    accept={ACCEPTED}
                    onChange={(e) => void handleFiles(e.target.files)}
                    disabled={uploading}
                  />
                  Browse files
                </label>
              </div>
            </div>
          </section>
        </div>

        <div className="upload-col upload-list">
          <section className="card">
            <div className="card-head">
              <h2>Uploaded documents</h2>
              <span className="chip">{documents.length}</span>
            </div>
            {documents.length === 0 ? (
              <p className="doc-empty">Nothing uploaded yet. Files appear here as soon as they are read.</p>
            ) : (
              <ul className="doc-rows">
                {documents.map((d) => {
                  // The live status wins while a poll is running: the row in
                  // `documents` was fetched when the upload returned, at which
                  // point every upload is "processing".
                  const live = progress[d.id];
                  const status = live?.status ?? d.status;
                  const pages = live?.page_count ?? d.page_count;
                  const fields = live?.field_count ?? d.fields.length;
                  const pillClass =
                    status === "extracted" ? "pill-extracted"
                    : status === "failed" ? "pill-failed"
                    : "pill-processing";

                  return (
                    <li className="doc-row" key={d.id}>
                      {status === "extracted" ? (
                        <img
                          className="doc-thumb"
                          src={renderDocumentUrl(d.id, 0)}
                          alt=""
                          onError={(e) => { e.currentTarget.style.visibility = "hidden"; }}
                        />
                      ) : (
                        <span className="doc-thumb doc-thumb-fallback" aria-hidden="true">
                          <FileIcon size={17} />
                        </span>
                      )}

                      <div className="doc-row-main">
                        <p className="doc-row-name" title={d.filename}>{d.filename}</p>
                        <p className="doc-row-meta">
                          {prettyType(d.doc_type)}
                          {d.extraction_source && ` · via ${d.extraction_source}`}
                          {pages != null && ` · ${pages} page${pages === 1 ? "" : "s"}`}
                          {status === "extracted" && ` · ${fields} field${fields === 1 ? "" : "s"}`}
                        </p>
                        {/* Only a document that actually FAILED says anything
                            here. `error` also carries advisory notes on
                            documents that extracted fine; those stay in the
                            API response and the review screen's JSON panel. */}
                        {status === "failed" && d.error && (
                          <p className="doc-row-meta" style={{ color: "var(--warn-ink)" }}>{d.error}</p>
                        )}
                        {live && !live.done && live.error && (
                          <p className="doc-row-meta" style={{ color: "var(--warn-ink)" }}>{live.error}</p>
                        )}
                      </div>

                      <div className="doc-row-actions">
                        <span className={`chip ${pillClass}`}>
                          {status === "extracted" && <CheckIcon size={13} />}
                          {status === "extracted" ? "Extracted" : status === "failed" ? "Failed" : "Processing"}
                        </span>
                        {status === "extracted" && (
                          <Link
                            className="link-review"
                            to={`/applications/${applicationId}/review?doc=${d.id}`}
                          >
                            Review
                          </Link>
                        )}
                        <button
                          className="btn-icon"
                          onClick={() => void handleRemove(d)}
                          aria-label={`Remove ${d.filename}`}
                          title="Remove"
                        >
                          <CloseIcon size={16} />
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </div>

        <aside className="upload-col upload-rail">
          <section className="card">
            <div className="card-head">
              <h2>What to upload</h2>
            </div>
            <div className="card-body">
              <ul className="rail-list">
                {CHECKLIST.map((item) => (
                  <li key={item}>
                    <CheckIcon size={15} />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <h2>Scan quality</h2>
              <GaugeIcon size={16} />
            </div>
            <div className="card-body">
              <p style={{ fontSize: 12.5, color: "var(--ink-2)", lineHeight: 1.5 }}>
                300 dpi or better, the whole page in frame and no glare. A skewed photograph of a
                calibration certificate is the single most common cause of a value coming back
                needing review.
              </p>
              <div className="rail-meter">
                <span style={{ width: `${Math.round((avgConfidence ?? 0) * 100)}%` }} />
              </div>
              <div className="rail-stat">
                <span className="micro">Average confidence</span>
                <span className="rail-stat-figure">
                  {avgConfidence === null ? "—" : `${Math.round(avgConfidence * 100)}%`}
                </span>
              </div>
            </div>
          </section>

          <section className="card-dark">
            <span className="card-dark-icon">
              <ShieldIcon size={16} />
            </span>
            <h2>Processed in region</h2>
            <p>
              Documents and extracted values stay on in-region infrastructure. Every upload,
              edit and confirmation is written to an audit trail against this application, so an
              assessor can see who confirmed which value and when.
            </p>
          </section>
        </aside>
      </div>

      <div className="sticky-bar">
        <p className="sticky-bar-summary">
          <strong>{documents.length}</strong> file{documents.length === 1 ? "" : "s"} uploaded ·{" "}
          <strong>{extracted.length}</strong> extracted · <strong>{totalFields}</strong> field
          {totalFields === 1 ? "" : "s"} ready to review
        </p>
        <div className="sticky-bar-actions">
          <button className="btn" onClick={refresh}>
            Save draft
          </button>
          <button
            className="btn btn-primary"
            disabled={extracted.length === 0}
            onClick={() => navigate(`/applications/${applicationId}/review`)}
          >
            Continue to review
            <span aria-hidden="true">→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
