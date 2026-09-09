import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import "../styles/graph.css";
import ExtractionGroups from "../components/ExtractionGroups";
import {
  getResult,
  getStatus,
  listRuns,
  retryDocument,
  submitDocument,
  submitReview,
  type FinalStatus,
  type GraphConflict,
  type GraphResult,
  type GraphStatus,
  type RunSummary,
} from "../api/graphClient";

/**
 * The agentic extraction screen.
 *
 * Three states, in order: submit a document, watch it process, read the result.
 * Progress is polled from the workflow's checkpoint rather than streamed —
 * a long document takes minutes and a poll costs nothing, while a held-open
 * connection is a liability on both ends.
 */

const NODE_LABELS: Record<string, string> = {
  start: "Starting",
  document_validation: "Checking the file",
  preprocessing: "Reading pages & OCR",
  document_analysis: "Analysing structure",
  chunk_management: "Splitting into chunks",
  dynamic_extraction: "Extracting fields",
  response_validation: "Validating responses",
  retry_recovery: "Retrying failed chunks",
  evidence_validation: "Checking evidence",
  field_normalization: "Normalising & merging",
  conflict_resolution: "Resolving conflicts",
  completeness_check: "Checking completeness",
  targeted_reanalysis: "Re-reading thin pages",
  form_mapping: "Mapping to the form",
  form_filling: "Filling the form",
  human_review: "Waiting for review",
  quality_control: "Quality control",
  final_decision: "Deciding",
  persistence: "Saving",
};

const STATUS_TONE: Record<FinalStatus, string> = {
  VALIDATED: "ok",
  LOW_CONFIDENCE: "warn",
  CONFLICTED: "warn",
  MANUAL_REVIEW_REQUIRED: "warn",
  INCOMPLETE: "bad",
  REJECTED: "bad",
};

const STATUS_HELP: Record<FinalStatus, string> = {
  VALIDATED: "Every page was read, every value is backed by the document.",
  LOW_CONFIDENCE: "Complete and consistent, but too little is firmly verified to use unchecked.",
  CONFLICTED: "The document states different values for the same field. Every candidate is kept below.",
  MANUAL_REVIEW_REQUIRED: "Usable, but something needs a person to decide.",
  INCOMPLETE: "Part of the document is unaccounted for. Do not treat this result as complete.",
  REJECTED: "The document could not be processed at all.",
};

function Check({ ok, label }: { ok: boolean; label: string }) {
  return (
    <li className={ok ? "qc-ok" : "qc-fail"}>
      <span aria-hidden="true">{ok ? "✓" : "✕"}</span> {label}
    </li>
  );
}

function ConflictCard({
  conflict,
  chosen,
  onChoose,
}: {
  conflict: GraphConflict;
  chosen: string | undefined;
  onChoose: (uid: string) => void;
}) {
  return (
    <div className="cf-card">
      <div className="cf-head">
        <strong>{conflict.normalized_field_name}</strong>
        <span className={`xf-badge tone-${conflict.resolution === "RESOLVED" ? "ok" : "warn"}`}>
          {conflict.resolution.replace(/_/g, " ").toLowerCase()}
        </span>
      </div>
      {conflict.reason && <p className="cf-reason">{conflict.reason}</p>}
      <div className="cf-options">
        {conflict.candidates.map((c) => (
          <label key={c.field_uid} className={`cf-option ${chosen === c.field_uid ? "picked" : ""}`}>
            <input
              type="radio"
              name={`cf-${conflict.normalized_field_name}`}
              checked={chosen === c.field_uid}
              onChange={() => onChoose(c.field_uid)}
            />
            <span className="cf-value">{c.value ?? "—"}</span>
            <span className="cf-where">
              page {c.page_number ?? "?"}
              {c.section_name && ` · ${c.section_name}`}
            </span>
            {c.evidence && <span className="cf-evidence">“{c.evidence}”</span>}
          </label>
        ))}
      </div>
    </div>
  );
}

export default function GraphExtractionPage() {
  const { documentId: routeDocumentId } = useParams();
  const [file, setFile] = useState<File | null>(null);
  const [formId, setFormId] = useState("");
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [status, setStatus] = useState<GraphStatus | null>(null);
  const [result, setResult] = useState<GraphResult | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [decisions, setDecisions] = useState<Record<string, string>>({});
  const [tab, setTab] = useState<"fields" | "form" | "review" | "audit">("fields");
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const refreshRuns = useCallback(() => {
    listRuns(15)
      .then((r) => setRuns(r.runs))
      .catch(() => {
        /* the list is a convenience; a failure here must not blank the page */
      });
  }, []);

  useEffect(() => {
    refreshRuns();
    return stopPolling;
  }, [refreshRuns, stopPolling]);


  const loadResult = useCallback(
    async (id: string) => {
      try {
        const r = await getResult(id);
        setResult(r);
        refreshRuns();
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e));
      }
    },
    [refreshRuns],
  );

  const startPolling = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = window.setInterval(async () => {
        try {
          const s = await getStatus(id);
          setStatus(s);
          const done = s.final_status != null || s.source === "database";
          if (done) {
            stopPolling();
            await loadResult(id);
          }
        } catch (e) {
          stopPolling();
          setError(String(e instanceof Error ? e.message : e));
        }
      }, 1500);
    },
    [loadResult, stopPolling],
  );

  // Deep link: /extraction/<id> opens that run. If it is still in flight the
  // result endpoint 409s, so fall back to polling rather than showing an error
  // for a document that is simply not finished yet.
  useEffect(() => {
    if (!routeDocumentId) return;
    setDocumentId(routeDocumentId);
    getResult(routeDocumentId)
      .then(setResult)
      .catch(() => startPolling(routeDocumentId));
  }, [routeDocumentId, startPolling]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setStatus(null);
    setDecisions({});
    try {
      const { document_id } = await submitDocument(file, { formId: formId || undefined });
      setDocumentId(document_id);
      startPolling(document_id);
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setBusy(false);
    }
  }

  async function openRun(id: string) {
    stopPolling();
    setDocumentId(id);
    setStatus(null);
    setResult(null);
    setDecisions({});
    setError(null);
    await loadResult(id);
  }

  async function applyDecisions() {
    if (!documentId || Object.keys(decisions).length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await submitReview(
        documentId,
        Object.entries(decisions).map(([field_name, chosen_field_uid]) => ({ field_name, chosen_field_uid })),
      );
      setResult(updated);
      setDecisions({});
      refreshRuns();
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setBusy(false);
    }
  }

  async function onRetry() {
    if (!documentId) return;
    setBusy(true);
    try {
      const updated = await retryDocument(documentId);
      setResult(updated);
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setBusy(false);
    }
  }

  const running = documentId != null && result == null;
  const qc = result?.quality_control;

  return (
    <div className="gx">
      <header className="gx-header">
        <div>
          <h1>Document extraction</h1>
          <p className="gx-sub">
            Reads every page, keeps a source quote for every value, and tells you what it could not
            verify rather than guessing.
          </p>
        </div>
      </header>

      {error && (
        <div className="gx-error" role="alert">
          <strong>Something went wrong.</strong> {error}
        </div>
      )}

      <div className="gx-layout">
        <aside className="gx-side">
          <form className="gx-panel" onSubmit={onSubmit}>
            <h2>New document</h2>
            <label className="gx-file">
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.docx"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <span>{file ? file.name : "Choose a PDF, image or DOCX…"}</span>
            </label>
            <label className="gx-field">
              <span>Target form (optional)</span>
              <input
                type="text"
                placeholder="e.g. NABL_151"
                value={formId}
                onChange={(e) => setFormId(e.target.value.trim())}
              />
            </label>
            <button type="submit" className="gx-primary" disabled={!file || busy}>
              {busy ? "Submitting…" : "Extract"}
            </button>
          </form>

          <div className="gx-panel">
            <h2>Recent</h2>
            {runs.length === 0 ? (
              <p className="xf-empty">Nothing processed yet.</p>
            ) : (
              <ul className="gx-runs">
                {runs.map((r) => (
                  <li key={r.document_id}>
                    <button
                      type="button"
                      className={documentId === r.document_id ? "active" : ""}
                      onClick={() => openRun(r.document_id)}
                    >
                      <span className="gx-run-name">{r.file_name || r.document_id}</span>
                      <span className={`xf-badge tone-${STATUS_TONE[r.final_status as FinalStatus] ?? "muted"}`}>
                        {(r.final_status || r.current_status || "").replace(/_/g, " ").toLowerCase()}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        <main className="gx-main">
          {running && (
            <div className="gx-panel gx-progress">
              <h2>Processing</h2>
              <p className="gx-node">{NODE_LABELS[status?.current_node ?? "start"] ?? status?.current_node ?? "Starting…"}</p>
              <div className="gx-bar" role="progressbar" aria-label="Chunks processed">
                <div
                  className="gx-bar-fill"
                  style={{
                    width: `${
                      status?.chunks_total
                        ? Math.round(((status.chunks_processed ?? 0) / status.chunks_total) * 100)
                        : 6
                    }%`,
                  }}
                />
              </div>
              <dl className="gx-stats">
                <div><dt>Pages</dt><dd>{status?.total_pages ?? "—"}</dd></div>
                <div><dt>Chunks</dt><dd>{status?.chunks_processed ?? 0}/{status?.chunks_total ?? "—"}</dd></div>
                <div><dt>Fields</dt><dd>{status?.fields_extracted ?? 0}</dd></div>
                <div><dt>Model calls</dt><dd>{status?.llm_calls ?? 0}</dd></div>
              </dl>
              {(status?.chunks_failed ?? 0) > 0 && (
                <p className="gx-warn">{status?.chunks_failed} chunk(s) failed and are being retried.</p>
              )}
            </div>
          )}

          {result && (
            <>
              <div className={`gx-verdict tone-${STATUS_TONE[result.overall_status]}`}>
                <div className="gx-verdict-head">
                  <span className="gx-verdict-status">{result.overall_status.replace(/_/g, " ")}</span>
                  <span className="gx-verdict-help">{STATUS_HELP[result.overall_status]}</span>
                </div>
                <p className="gx-verdict-reason">{result.final_reasoning}</p>
                <dl className="gx-stats">
                  <div><dt>Pages</dt><dd>{result.document_summary.page_count}</dd></div>
                  <div><dt>OCR pages</dt><dd>{result.document_summary.ocr_pages}</dd></div>
                  <div><dt>Fields</dt><dd>{result.extracted_fields.length}</dd></div>
                  <div><dt>Failed chunks</dt><dd>{result.document_summary.failed_chunks}</dd></div>
                  <div><dt>Model calls</dt><dd>{result.metrics.llm_call_count}</dd></div>
                  <div><dt>Time</dt><dd>{result.metrics.processing_time_seconds}s</dd></div>
                </dl>
                {qc && (
                  <ul className="gx-qc">
                    <Check ok={qc.all_pages_processed} label="every page processed" />
                    <Check ok={qc.all_chunks_processed} label="every chunk processed" />
                    <Check ok={qc.all_values_have_evidence} label="every value evidenced" />
                    <Check ok={qc.conflicts_resolved} label="conflicts resolved" />
                    <Check ok={!qc.information_loss_detected} label="no information loss" />
                  </ul>
                )}
                {result.document_summary.failed_chunks > 0 && (
                  <button type="button" className="gx-secondary" onClick={onRetry} disabled={busy}>
                    Retry failed chunks
                  </button>
                )}
              </div>

              <nav className="gx-tabs">
                {(["fields", "form", "review", "audit"] as const).map((t) => (
                  <button key={t} type="button" className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                    {t === "fields" && `Extracted (${result.extracted_fields.length})`}
                    {t === "form" && `Form (${result.form_mappings.length})`}
                    {t === "review" && `Review (${result.manual_review_items.length + result.conflicts.length})`}
                    {t === "audit" && "Audit"}
                  </button>
                ))}
              </nav>

              <div className="gx-panel">
                {tab === "fields" && <ExtractionGroups fields={result.extracted_fields} />}

                {tab === "form" && (
                  <>
                    <h2>Target form</h2>
                    {result.form_mappings.length === 0 ? (
                      <p className="xf-empty">No target form was supplied, so nothing was mapped.</p>
                    ) : (
                      <table className="gx-table">
                        <thead>
                          <tr>
                            <th>Form field</th>
                            <th>Value</th>
                            <th>From</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.form_mappings.map((m) => (
                            <tr key={m.target_field}>
                              <td className="mono">{m.target_field}</td>
                              <td>{m.mapped_value ?? <em className="xf-null">not filled</em>}</td>
                              <td title={m.mapping_reason}>
                                {m.source_field || "—"}
                                {m.source_page != null && <span className="xf-page-flat"> p{m.source_page}</span>}
                              </td>
                              <td>
                                <span
                                  className={`xf-badge tone-${
                                    m.mapping_status === "MAPPED"
                                      ? "ok"
                                      : m.mapping_status === "NOT_FOUND"
                                        ? "muted"
                                        : "warn"
                                  }`}
                                >
                                  {m.mapping_status.replace(/_/g, " ").toLowerCase()}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </>
                )}

                {tab === "review" && (
                  <>
                    <h2>Needs a decision</h2>
                    {result.conflicts.length === 0 && result.manual_review_items.length === 0 ? (
                      <p className="xf-empty">Nothing is waiting on you.</p>
                    ) : (
                      <>
                        {result.conflicts.map((c) => (
                          <ConflictCard
                            key={c.normalized_field_name}
                            conflict={c}
                            chosen={decisions[c.normalized_field_name] ?? c.chosen_field_uid ?? undefined}
                            onChoose={(uid) =>
                              setDecisions((d) => ({ ...d, [c.normalized_field_name]: uid }))
                            }
                          />
                        ))}
                        {result.manual_review_items
                          .filter((i) => i.kind !== "conflict")
                          .map((i, idx) => (
                            <div className="cf-card" key={`${i.kind}-${i.reference}-${idx}`}>
                              <div className="cf-head">
                                <strong>{i.reference}</strong>
                                <span className="xf-badge tone-warn">{i.kind.replace(/_/g, " ")}</span>
                              </div>
                              <p className="cf-reason">{i.summary}</p>
                              {i.suggested_action && <p className="cf-action">{i.suggested_action}</p>}
                            </div>
                          ))}
                        {Object.keys(decisions).length > 0 && (
                          <button type="button" className="gx-primary" onClick={applyDecisions} disabled={busy}>
                            Apply {Object.keys(decisions).length} decision(s) and continue
                          </button>
                        )}
                      </>
                    )}
                  </>
                )}

                {tab === "audit" && (
                  <>
                    <h2>What happened</h2>
                    <ol className="gx-audit">
                      {result.audit_log.map((a, i) => (
                        <li key={i} className={`lvl-${a.level}`}>
                          <span className="gx-audit-node">{a.node}</span>
                          <span className="gx-audit-event">{a.event.replace(/_/g, " ")}</span>
                          <span className="gx-audit-detail">{a.detail}</span>
                        </li>
                      ))}
                    </ol>
                  </>
                )}
              </div>
            </>
          )}

          {!running && !result && (
            <div className="gx-panel gx-idle">
              <h2>Nothing loaded</h2>
              <p>Submit a document on the left, or open one of the recent runs.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
