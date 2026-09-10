import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  createApplication,
  listDocuments,
  uploadDocument,
  type DocumentOut,
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
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    setUploading(true);
    setError(null);
    try {
      for (const file of Array.from(files)) {
        const doc = await uploadDocument(applicationId, file, script);
        setDocuments((docs) => [...docs.filter((d) => d.id !== doc.id), doc]);
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
        {documents.map((d) => (
          <li key={d.id} className={`document-status-${d.status}`}>
            <strong>{d.filename}</strong> — {d.status}
            {d.doc_type && <span> · {d.doc_type}</span>}
            {d.extraction_source && <span> · via {d.extraction_source}</span>}
            {/* Only a document that actually FAILED says anything here.
                `error` also carries advisory extraction notes on documents
                that extracted fine ("3 values could not be quoted", "6 items
                need a person to decide"); those are deliberately not
                surfaced in the UI — they remain in the API response and in
                the review screen's "View JSON". */}
            {d.error && d.status === "failed" && <span className="error"> · {d.error}</span>}
          </li>
        ))}
      </ul>

      <button className="primary" disabled={!anyExtracted} onClick={() => navigate(`/applications/${applicationId}/review`)}>
        Continue to review →
      </button>
    </div>
  );
}
