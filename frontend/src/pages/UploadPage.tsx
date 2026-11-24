import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  createApplication,
  getDocumentStatus,
  listDocuments,
  uploadDocument,
  type DocumentOut,
  type DocumentStatus,
  type NablFormType,
} from "../api/client";

/**
 * Upload — the entry point.
 *
 * There is no wizard and no start step. Landing here with no application in
 * the URL silently gets one and shows the upload box; the id is remembered so
 * a reload does not orphan documents already uploaded.
 *
 * The application is created ALREADY UNLOCKED, which is load-bearing: the
 * upload endpoint refuses anything still in "eligibility" status, and that is
 * the gate the wizard used to satisfy.
 */

const SCRIPTS = [
  { value: "english", label: "English (Latin)" },
  { value: "devanagari", label: "Devanagari" },
  { value: "arabic", label: "Arabic" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "ka", label: "Kannada" },
];

//: Which NABL form the compiled output targets. Not asked for on screen —
//: there is no "start" step — so it lives here until the product needs a
//: picker again.
const DEFAULT_FORM_TYPE: NablFormType = "NABL_151";

const POLL_INTERVAL_MS = 2000;
//: A 17-page PDF has measured well over 20 minutes end to end, so the ceiling
//: is clear of that. Reaching it means "still running", not "failed".
const POLL_TIMEOUT_MS = 30 * 60 * 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

//: Per-document wall-clock timer: starts when the file is picked (so it
//: includes the upload itself), stops when the server reports the document
//: done — extracted or failed. `end` stays unset while still running.
type Timer = { start: number; end?: number };

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

//: Deliberately NOT remembered across visits. Returning to the upload screen
//: to add a NEW document used to reuse the previous application and list its
//: documents, so the last file was still sitting there — it read as though the
//: upload had not worked, or worse, as though the new document had already
//: been processed.
//:
//: The URL is the source of truth instead. "/" always starts a fresh
//: application; "/applications/<id>/upload" carries its own id, so reloading
//: that page still shows exactly the documents belonging to it.

export default function UploadPage() {
  const { applicationId } = useParams();
  const navigate = useNavigate();
  const [starting, setStarting] = useState(false);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [script, setScript] = useState("english");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, DocumentStatus>>({});
  const [timers, setTimers] = useState<Record<string, Timer>>({});
  const [now, setNow] = useState(() => Date.now());
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Tick once a second, but only while some timer is still running.
  const anyTimerRunning = Object.values(timers).some((t) => t.end === undefined);
  useEffect(() => {
    if (!anyTimerRunning) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [anyTimerRunning]);
  // Set on unmount, so a poll loop started here stops calling setState once
  // the user has navigated to the review screen.
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

  /**
   * Wait for one document to finish extracting.
   *
   * Bounded: a stuck job must not poll forever. On timeout the document is
   * left in whatever state the server reports — it is still processing, and
   * reloading picks it up — rather than being shown as failed, which would
   * be a lie.
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
        // with the very extraction being polled. Retry on the next tick.
        await sleep(POLL_INTERVAL_MS);
        continue;
      }

      setProgress((current) => ({ ...current, [documentId]: status }));
      if (status.done) {
        const finishedAt = Date.now();
        setTimers((current) => current[documentId]
          ? { ...current, [documentId]: { ...current[documentId], end: finishedAt } }
          : current);
        return;
      }
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
        error: `Still processing after ${Math.round(POLL_TIMEOUT_MS / 60000)} minutes. `
             + "It has not failed — reload this page to pick up the result.",
      },
    }));
  }, []);

  // Nothing in the URL means the user landed on "/" — get an application and
  // move straight to the upload box rather than asking them to start one.
  useEffect(() => {
    if (!applicationId && !starting && !error) {
      void start();
    }
    // `starting`/`error` deliberately excluded: this must fire once, not
    // retry in a loop when creation fails.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applicationId, start]);

  async function handleFiles(files: FileList | null) {
    if (!applicationId || !files || files.length === 0) return;
    // A second submission while one is in flight costs a full extraction.
    if (uploading) return;

    setUploading(true);
    setError(null);
    cancelledRef.current = false;
    try {
      for (const file of Array.from(files)) {
        const startedAt = Date.now();
        setNow(startedAt);
        const doc = await uploadDocument(applicationId, file, script);
        setTimers((current) => ({ ...current, [doc.id]: { start: startedAt } }));
        setDocuments((docs) => [...docs.filter((d) => d.id !== doc.id), doc]);
        // Upload answers 202 the moment the row exists and extracts
        // afterwards, so the document arrives here as "processing" and would
        // stay that way on screen — leaving "Continue to review" disabled
        // forever — unless someone asks the server when it has finished.
        //
        // Sequential on purpose: each extraction is CPU- and LLM-bound, so
        // starting several at once makes them all slower and increases the
        // chance of provider throttling.
        await waitForExtraction(doc.id);
        refresh();
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  // ---- no application in the URL: get one and carry on -------------------
  // The application is bookkeeping the user did not ask for, so it is created
  // silently and remembered, and the first thing they see is the upload box.
  if (!applicationId) {
    return (
      <div className="page upload-page">
        <h1>Upload supporting documents</h1>
        <p>{error ? error : "Preparing…"}</p>
        {error && (
          <button className="primary" onClick={start} disabled={starting}>
            Try again
          </button>
        )}
      </div>
    );
  }

  const anyExtracted = documents.some((d) => d.status === "extracted");

  return (
    <div className="page upload-page">
      <h1>Upload supporting documents</h1>
      <p>Legal proofs (GST/CIN), equipment calibration certificates, personnel CVs, PT/ILC results, SOP excerpts.</p>

      <label>
        Script of scanned/photo pages:
        <select value={script} onChange={(e) => setScript(e.target.value)}>
          {SCRIPTS.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
      </label>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.jpg,.jpeg,.png"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={uploading}
      />
      {uploading && <p>Uploading & extracting…</p>}
      {error && <p className="error">{error}</p>}

      <ul className="document-list">
        {documents.map((d) => {
          // The polled status wins while an upload is in flight: the row in
          // `documents` was fetched when upload returned 202, at which point
          // every document is "processing". Without this the list would sit
          // on that first snapshot until the whole batch finished.
          const live = progress[d.id];
          const status = live?.status ?? d.status;
          const pages = live?.page_count ?? d.page_count;
          const fields = live?.field_count;
          const timer = timers[d.id];
          return (
            <li key={d.id} className={`document-status-${status}`}>
              <strong>{d.filename}</strong> — {status}
              {timer && (
                <span className="upload-timer">
                  {" · "}
                  {timer.end === undefined
                    ? `⏱ ${formatElapsed(now - timer.start)}`
                    : `took ${formatElapsed(timer.end - timer.start)}`}
                </span>
              )}
              {pages != null && <span> · {pages} page{pages === 1 ? "" : "s"}</span>}
              {fields != null && status === "extracted" && (
                <span> · {fields} field{fields === 1 ? "" : "s"}</span>
              )}
              {d.doc_type && <span> · {d.doc_type}</span>}
              {d.extraction_source && <span> · via {d.extraction_source}</span>}
              {/* Only a document that actually FAILED says anything here.
                  `error` also carries advisory extraction notes on documents
                  that extracted fine ("3 values could not be quoted", "6 items
                  need a person to decide"); those are deliberately not
                  surfaced in the UI — they remain in the API response. */}
              {d.error && status === "failed" && <span className="error"> · {d.error}</span>}
              {/* A poll that ran out of time is not a failure: the job is
                  still going server-side, so say so. */}
              {live && !live.done && live.error && (
                <span className="warn"> · {live.error}</span>
              )}
            </li>
          );
        })}
      </ul>

      <button className="primary" disabled={!anyExtracted} onClick={() => navigate(`/applications/${applicationId}/review`)}>
        Continue to review →
      </button>
    </div>
  );
}
