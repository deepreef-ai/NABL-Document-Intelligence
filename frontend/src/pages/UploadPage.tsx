import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { listDocuments, uploadDocument, type DocumentOut } from "../api/client";

const SCRIPTS = [
  { value: "english", label: "English (Latin)" },
  { value: "devanagari", label: "Devanagari" },
  { value: "arabic", label: "Arabic" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "ka", label: "Kannada" },
];

export default function UploadPage() {
  const { applicationId } = useParams();
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [script, setScript] = useState("english");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function refresh() {
    if (!applicationId) return;
    listDocuments(applicationId).then(setDocuments).catch((e) => setError(String(e)));
  }

  useEffect(refresh, [applicationId]);

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
            {d.error && <span className="error"> · {d.error}</span>}
          </li>
        ))}
      </ul>

      <button className="primary" disabled={!anyExtracted} onClick={() => navigate(`/applications/${applicationId}/review`)}>
        Continue to review →
      </button>
    </div>
  );
}
