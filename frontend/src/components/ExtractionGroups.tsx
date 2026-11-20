import { useMemo, useState } from "react";
import {
  groupFields,
  type EvidenceStatus,
  type ExtractionStatus,
  type FieldGroup,
  type GraphField,
} from "../api/graphClient";

/**
 * Extracted fields, shown the way the document is organised: a heading, then
 * the tests underneath it.
 *
 * A flat list of forty fields throws away the structure the document had, and
 * that structure is what makes the values make sense together — "4.2 %" means
 * something under "Results" and nothing on its own. Table rows nest a second
 * level so twelve equipment rows read as one table with twelve rows rather
 * than twelve unrelated fields sharing a name.
 */

function statusTone(status: ExtractionStatus): string {
  switch (status) {
    case "VERIFIED":
      return "ok";
    case "CONFLICTED":
      return "bad";
    case "NOT_FOUND":
      return "muted";
    default:
      return "warn";
  }
}

function evidenceLabel(status: EvidenceStatus | null): { text: string; tone: string; title: string } {
  switch (status) {
    case "SUPPORTED":
      return { text: "evidenced", tone: "ok", title: "The quoted text was found on the cited page and contains this value." };
    case "PARTIALLY_SUPPORTED":
      return { text: "partial", tone: "warn", title: "The quote is real but does not contain the extracted value." };
    case "WRONG_PAGE":
      return { text: "page fixed", tone: "warn", title: "The quote was found on a different page; the page number was corrected." };
    case "UNSUPPORTED":
      return { text: "unsupported", tone: "bad", title: "The quoted text does not appear anywhere in the document. Value withheld." };
    case "NO_EVIDENCE":
      return { text: "no quote", tone: "bad", title: "No usable source quote was returned for this value." };
    default:
      return { text: "unchecked", tone: "muted", title: "Evidence has not been validated." };
  }
}

function FieldRow({ field, onFocusPage }: { field: GraphField; onFocusPage?: (page: number) => void }) {
  const [open, setOpen] = useState(false);
  const evidence = evidenceLabel(field.evidence_status);
  const hasDetail = Boolean(field.exact_source_evidence || field.notes || field.merged_from.length);

  return (
    <div className={`xf-row xf-${statusTone(field.extraction_status)}`}>
      <button
        type="button"
        className="xf-row-main"
        onClick={() => hasDetail && setOpen((v) => !v)}
        aria-expanded={hasDetail ? open : undefined}
        disabled={!hasDetail}
      >
        <span className="xf-name" title={field.field_name}>
          {field.field_name}
          {field.occurrence_index > 0 && <span className="xf-occ">#{field.occurrence_index + 1}</span>}
        </span>

        <span className={`xf-value ${field.value == null ? "xf-null" : ""}`}>
          {field.value ?? "— not stated —"}
        </span>

        <span className="xf-meta">
          {field.page_number != null && (
            <button
              type="button"
              className="xf-page"
              onClick={(e) => {
                e.stopPropagation();
                onFocusPage?.(field.page_number as number);
              }}
              title={`Go to page ${field.page_number}`}
            >
              p{field.page_number}
            </button>
          )}
          <span className={`xf-badge tone-${evidence.tone}`} title={evidence.title}>
            {evidence.text}
          </span>
          <span className="xf-conf" title="How clearly the source states this value">
            {Math.round(field.confidence_score * 100)}%
          </span>
        </span>
      </button>

      {open && hasDetail && (
        <div className="xf-detail">
          {field.exact_source_evidence && (
            <blockquote className="xf-quote">
              <span className="xf-quote-label">source</span>
              {field.exact_source_evidence}
            </blockquote>
          )}
          <dl className="xf-detail-grid">
            <dt>normalised</dt>
            <dd>{field.normalized_field_name}</dd>
            {field.normalized_value != null && field.normalized_value !== field.value && (
              <>
                <dt>value as stored</dt>
                <dd>{field.normalized_value}</dd>
              </>
            )}
            <dt>type</dt>
            <dd>{field.data_type}</dd>
            {field.merged_from.length > 0 && (
              <>
                <dt>merged from</dt>
                <dd>{field.merged_from.join(", ")}</dd>
              </>
            )}
            {field.notes && (
              <>
                <dt>notes</dt>
                <dd>{field.notes}</dd>
              </>
            )}
          </dl>
        </div>
      )}
    </div>
  );
}

function Group({
  group,
  defaultOpen,
  onFocusPage,
}: {
  group: FieldGroup;
  defaultOpen: boolean;
  onFocusPage?: (page: number) => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const pageLabel =
    group.pages.length === 0
      ? ""
      : group.pages.length === 1
        ? `page ${group.pages[0]}`
        : `pages ${group.pages[0]}–${group.pages[group.pages.length - 1]}`;

  return (
    <section className="xf-group">
      <button type="button" className="xf-heading" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className={`xf-caret ${open ? "open" : ""}`} aria-hidden="true">
          ▸
        </span>
        <span className="xf-heading-text">{group.heading}</span>
        <span className="xf-heading-meta">
          {pageLabel && <span className="xf-heading-pages">{pageLabel}</span>}
          <span className="xf-count">
            {group.verified}/{group.total} verified
          </span>
        </span>
      </button>

      {open && (
        <div className="xf-group-body">
          {group.rows.map((r) => (
            <FieldRow key={r.field.field_uid} field={r.field} onFocusPage={onFocusPage} />
          ))}

          {group.tables.map((table) => (
            <div className="xf-table" key={table.context}>
              <div className="xf-table-caption" title="A repeated row from a table in the document">
                {table.context}
                <span className="xf-count">{table.rows.length} value(s)</span>
              </div>
              {table.rows.map((r) => (
                <FieldRow key={r.field.field_uid} field={r.field} onFocusPage={onFocusPage} />
              ))}
            </div>
          ))}

          {group.rows.length === 0 && group.tables.length === 0 && (
            <p className="xf-empty">Nothing was extracted under this heading.</p>
          )}
        </div>
      )}
    </section>
  );
}

export default function ExtractionGroups({
  fields,
  onFocusPage,
}: {
  fields: GraphField[];
  onFocusPage?: (page: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [onlyProblems, setOnlyProblems] = useState(false);

  const groups = useMemo(() => {
    let visible = fields;
    if (onlyProblems) {
      visible = visible.filter(
        (f) =>
          f.extraction_status !== "VERIFIED" ||
          (f.evidence_status != null && f.evidence_status !== "SUPPORTED"),
      );
    }
    const q = query.trim().toLowerCase();
    if (q) {
      visible = visible.filter(
        (f) =>
          f.field_name.toLowerCase().includes(q) ||
          (f.value ?? "").toLowerCase().includes(q) ||
          f.section_name.toLowerCase().includes(q),
      );
    }
    return groupFields(visible);
  }, [fields, query, onlyProblems]);

  const shown = groups.reduce((n, g) => n + g.total, 0);

  if (fields.length === 0) {
    return <p className="xf-empty">No fields were extracted from this document.</p>;
  }

  return (
    <div className="xf">
      <div className="xf-toolbar">
        <input
          className="xf-search"
          type="search"
          placeholder="Filter by heading, field or value…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Filter extracted fields"
        />
        <label className="xf-toggle">
          <input type="checkbox" checked={onlyProblems} onChange={(e) => setOnlyProblems(e.target.checked)} />
          Needs attention only
        </label>
        <span className="xf-toolbar-count">
          {shown} of {fields.length} field(s) · {groups.length} heading(s)
        </span>
      </div>

      {groups.length === 0 ? (
        <p className="xf-empty">Nothing matches that filter.</p>
      ) : (
        // The first heading opens by default so the page is useful at rest;
        // opening all of them on a 40-heading document would bury the reader.
        groups.map((g, i) => (
          <Group key={g.heading} group={g} defaultOpen={i === 0 || groups.length <= 3} onFocusPage={onFocusPage} />
        ))
      )}
    </div>
  );
}
